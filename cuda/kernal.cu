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

#define PARTICLE_COUNT 50

sphere spheres[PARTICLE_COUNT];
uint threadPerBlock = 25;
uint blocks = PARTICLE_COUNT / threadPerBlock;

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ vec3 castRay(const ray &r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    } else {
//        vec3 unit_direction = unit_vector(r.direction());
//        float t = 0.5f * (unit_direction.y() + 1.0f);
//        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
        return vec3(0, 0, 0);
    }
}

__global__ void d_render(uchar4 *d_output, uint width, uint height, hitable **d_world) {
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
        vec3 col = castRay(r, d_world);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }

}

extern "C"
void onIdle() {
    for (auto &sphere : spheres) {
        sphere.move(0.005);
    }
}

__global__ void create_world(sphere *d_spheres, hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

//    int i = threadIdx.x + (blockDim.x * blockIdx.x);

        for (int i = 0; i < PARTICLE_COUNT; i++) {
            auto s = d_spheres[i];
//            *(d_list + i) = &d_spheres[i];

//            printf("%f\n", s.position.x());

            *(d_list + i) = new sphere(s);
        }

        *d_world = new hitable_list(d_list, PARTICLE_COUNT);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < PARTICLE_COUNT; i++) {
            delete *(d_list + i);
        }
        delete *d_world;
    }
}

__global__ void move_particles(sphere *d_spheres) {
//    int i = threadIdx.x + (blockDim.x * blockIdx.x);
//    d_spheres[i].move(1);
}

extern "C"
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output) {
    sphere *d_particleList;
    checkCudaErrors(cudaMalloc((void **) &d_particleList, PARTICLE_COUNT * sizeof(sphere)));
    checkCudaErrors(cudaMemcpy(d_particleList, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

    // make our world of hitables
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **) &d_list, 2 * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hitable *)));

    create_world <<< blocks, threadPerBlock >>>(d_particleList, d_list, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

//    move_particles <<< blocks, threadPerBlock >>>(d_particleList);
//
//    checkCudaErrors(cudaGetLastError());
//    checkCudaErrors(cudaDeviceSynchronize());

    // call CUDA kernel, writing results to PBO memory
    d_render <<< gridSize, blockSize >>>(output, width, height, d_world);
    getLastCudaError("kernel failed");

    free_world<<< blocks, threadPerBlock >>>(d_list, d_world);
    getLastCudaError("kernel failed");

    checkCudaErrors(cudaMemcpy(spheres, d_particleList, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyDeviceToHost));
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

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
