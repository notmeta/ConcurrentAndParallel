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

// My Files
#include "vec3.h"
#include "sphere.h"
#include "Ray.h"
#include "hitable_list.h"
#include "hitable.h"
#include "random"

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

#define PARTICLE_COUNT 500

sphere spheres[PARTICLE_COUNT];
uint threadPerBlock = 25;
uint blocks = PARTICLE_COUNT / threadPerBlock;

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ vec3 castRay(const ray &r, sphere *d_spheres) {
    hit_record rec;

    bool hit_anything = false;
    float closest_so_far = FLT_MAX;
    for (int i = 0; i < PARTICLE_COUNT; i++) {
        if (d_spheres[i].hit(r, 0, closest_so_far, rec)) {
            hit_anything = true;
            closest_so_far = rec.t;
            rec = rec;
        }
    }

    if (hit_anything) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    } else {
//        vec3 unit_direction = unit_vector(r.direction());
//        float t = 0.5f * (unit_direction.y() + 1.0f);
//        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
        return vec3(0, 0, 0);
    }
}

__global__ void d_render(uchar4 *d_output, uint width, uint height, sphere *d_spheres) {
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
        vec3 col = castRay(r, d_spheres);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }

}

extern "C"
void onIdle() {
}

__global__ void move_particles(sphere *d_spheres) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    d_spheres[i].move(0.5);
}

extern "C"
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output) {
    sphere *d_particleList = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_particleList, PARTICLE_COUNT * sizeof(sphere)));
    checkCudaErrors(cudaMemcpy(d_particleList, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

    move_particles <<< blocks, threadPerBlock >>>(d_particleList);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    d_render <<< gridSize, blockSize >>>(output, width, height, d_particleList);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

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
