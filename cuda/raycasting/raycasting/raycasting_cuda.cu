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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

// #include "raycasting_kernel.cuh"

cudaArray *d_imageArray = 0;

extern "C"
void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_imageArray;
}

extern "C"
void freeTexture()
{
    checkCudaErrors(cudaFreeArray(d_imageArray));
}


__global__
void
d_render(uchar4	*d_output, uint width, uint height)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	if (x >= width || y >= height)
	{
		return;
	}

	if (x % 4 == 0)
	{
		d_output[i] = make_uchar4(255, 0, 0, 0);
	}
	
	if (y % 2 == 0)
	{
		d_output[i] = make_uchar4(0, 0, 255, 0);
	}

	if (x % 5 == 0)
	{
		d_output[i] = make_uchar4(0, 255, 0, 0);
	}

}

// render image using CUDA
extern "C"
void render(int width, int height,
            dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	d_render << <gridSize, blockSize >> > (output, width, height);
	// call CUDA kernel, writing results to PBO memory

    getLastCudaError("kernel failed");
}

#endif
