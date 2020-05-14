/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <cfloat>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "vec3.h"
#include "sphere.h"
#include "Ray.h"
#include "hitable.h"
#include "random"

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

#define PARTICLE_COUNT 200
#define BACKGROUND_COLOUR make_uchar4(0, 0, 0, 0)
#define PARTICLES_COLLIDE false

bool gravityEnabled = false;
sphere spheres[PARTICLE_COUNT];
ColourMode mode = SOLID;

uint threadPerBlock = 20;
uint blocks = PARTICLE_COUNT / threadPerBlock;

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ uchar4 castRay(const ray &r, sphere *d_spheres, ColourMode d_mode) {
    hit_record rec;
    uchar4 colour;
    bool hitSomething = false;
    float closestParticleSoFar = FLT_MAX;

    for (int i = 0; i < PARTICLE_COUNT; i++) {
        if (d_spheres[i].hit(r, 0, closestParticleSoFar, rec)) {
            colour = d_spheres[i].getColour(d_mode);
            hitSomething = true;
            closestParticleSoFar = rec.t;
            rec = rec;
        }
    }

    if (hitSomething) {
        return colour;
    } else {
        return BACKGROUND_COLOUR;
    }
}

__global__ void d_render(uchar4 *d_output, uint width, uint height, sphere *d_spheres, ColourMode d_mode) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;
    float u = x / (float) width; //----> [0, 1]x[0, 1]
    float v = y / (float) height;
    u = 2.0 * u - 1.0; //---> [-1, 1]x[-1, 1]
    v = -(2.0 * v - 1.0);
    u *= width / (float) height;
    u *= 2.0;
    v *= 2.0;
    vec3 eye = vec3(0, 0.5, 1.5);
    float distFrEye2Img = 1.0;;
    if ((x < width) && (y < height)) {
        //for each pixel
        vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
        //fire a ray:
        ray r;
        r.O = eye;
        r.Dir = pixelPos - eye;
        //view direction along negtive z-axis!
        d_output[i] = castRay(r, d_spheres, d_mode);
    }
}

extern "C"
void onIdle() {
}

extern "C"
void setGravity(bool enabled) {
    gravityEnabled = enabled;
}

extern "C"
void setColourMode(int cMode) {
    cMode -= 48;
    mode = static_cast<ColourMode>(cMode);
    printf("Setting colour mode to %d\n", mode);
}

__global__ void move_particles(sphere *d_spheres) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    d_spheres[i].move(0.5);
}

__global__ void applyGravity(sphere *d_spheres) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);

    auto v = d_spheres[i].velocity;
//    d_spheres[i].velocity -= vec3(0, 0.6, 0);
    d_spheres[i].velocity = vec3(0, -3, 0);
    d_spheres[i].move(0.5);
    d_spheres[i].velocity = v;
}

__global__ void handleCollisions(sphere *d_spheres) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);

    for (int j = 0; j < PARTICLE_COUNT; j++) {
        if (i == j) { // it's checking itself
            continue;
        }
        auto other = d_spheres[j];
        if (other.updated) {
            continue;
        }
        if (d_spheres[i].overlaps(&other)) {
            d_spheres[i].changeVelocities(&other);
            break;
        }
    }
}

extern "C"
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output) {
    sphere *d_particleList = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_particleList, PARTICLE_COUNT * sizeof(sphere)));
    checkCudaErrors(cudaMemcpy(d_particleList, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

    move_particles <<< blocks, threadPerBlock >>>(d_particleList);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    d_render <<< gridSize, blockSize >>>(output, width, height, d_particleList, mode);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (PARTICLES_COLLIDE) {
        handleCollisions <<< blocks, threadPerBlock >>>(d_particleList);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (gravityEnabled) {
        applyGravity <<< blocks, threadPerBlock >>>(d_particleList);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaMemcpy(spheres, d_particleList, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyDeviceToHost));
    getLastCudaError("kernel failed");
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#endif
