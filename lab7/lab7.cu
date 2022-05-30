#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "omp.h"
#include "DS_definitions.h"
#include "DS_timer.h"

#define BLOCK_SIZE 1024
#define NUM_THREADS 6

// lab7-1
// kernel fuction : w/o shared mem
__global__ void integral(float* c, float a, float b, float dx) {
    float x1 = a + (blockDim.x * blockIdx.x + threadIdx.x) * dx;
    float x2 = a + (blockDim.x * blockIdx.x + threadIdx.x + 1) * dx;

    if (b <= x1) {
        return;
    }

    atomicAdd(c, ((x1 * x1) + (x2 * x2)) / 2.0 * dx);
    __syncthreads();
}

// kernel fuction : w shared mem
__global__ void integralWShared(float* c, float a, float b, float dx) {
    float x1 = a + (blockDim.x * blockIdx.x + threadIdx.x) * dx;
    float x2 = a + (blockDim.x * blockIdx.x + threadIdx.x + 1) * dx;

    __shared__ float localSum[BLOCK_SIZE];

    if (b <= x1) {
        return;
    }

    localSum[threadIdx.x] = ((x1 * x1) + (x2 * x2)) / 2.0 * dx;
    __syncthreads();

    int offset = 1;
    while (offset < BLOCK_SIZE) {
        if (threadIdx.x % (2 * offset) == 0) {
            localSum[threadIdx.x] += localSum[threadIdx.x + offset];
        }
        __syncthreads();
        offset *= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(c, localSum[0]);
    }
}

// kernel fuction : w shared mem while reducing bank conflict
__global__ void integralReduceBank(float* c, float a, float b, float dx) {
    float x1 = a + (blockDim.x * blockIdx.x + threadIdx.x) * dx;
    float x2 = a + (blockDim.x * blockIdx.x + threadIdx.x + 1) * dx;

    __shared__ float localSum[BLOCK_SIZE];

    if (b <= x1) {
        return;
    }

    localSum[threadIdx.x] = ((x1 * x1) + (x2 * x2)) / 2.0 * dx;
    __syncthreads();

    int offset = BLOCK_SIZE / 2;
    while (offset >= 1) {
        if (threadIdx.x < offset) {
            localSum[threadIdx.x] += localSum[threadIdx.x + offset];
        }
        __syncthreads();
        offset /= 2;
    }

    if (threadIdx.x == 0) {
        atomicAdd(c, localSum[0]);
    }
}


int main()
{
    // Set timer
    DS_timer timer(4);
    timer.setTimerName(0, (char*)"Host_OpenMP");
    timer.setTimerName(1, (char*)"Device_kernel1");
    timer.setTimerName(2, (char*)"Device_kernel2");
    timer.setTimerName(3, (char*)"Device_kernel3");


    // init variables
    float a = -5.0;
    float b = 5.0;

    int n = 1024 * 1024;
    float dx = (b - a) / n;

    float hIntegral = 0.0;
    float dIntegral1 = 0.0;
    float dIntegral2 = 0.0;
    float dIntegral3 = 0.0;

    float* dC;

    // 1. Host OpenMP : Integral
    timer.onTimer(0);
#pragma omp parallel for reduction(+:hIntegral) num_threads(NUM_THREADS) shared(dx)
    for (int i = 0; i < n; ++i) {
        hIntegral += ((a + i * dx) * (a + i * dx) + (a + (i + 1) * dx) * (a + (i + 1) * dx)) / 2.0 * dx;
    }
    timer.offTimer(0);

    // 2. Device : Integral using Atomic operation
    // 2.1 set
    cudaMalloc(&dC, sizeof(float));
    cudaMemset(dC, 0, sizeof(float));

    // 2.2 kernel
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(int(ceil((float)n / BLOCK_SIZE)), 1, 1);

    timer.onTimer(1);
    integral << <dimGrid, dimBlock >> > (dC, a, b, dx);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    cudaMemcpy(&dIntegral1, dC, sizeof(float), cudaMemcpyDeviceToHost);


    // 3. Device : Integral using shared mem
    // 3.1 set
    cudaMemset(dC, 0, sizeof(float));

    // 3.2 kernel
    timer.onTimer(2);
    integralWShared << <dimGrid, dimBlock >> > (dC, a, b, dx);
    cudaDeviceSynchronize();
    timer.offTimer(2);

    cudaMemcpy(&dIntegral2, dC, sizeof(float), cudaMemcpyDeviceToHost);


    // 4. Device : kernel fuction : w shared mem while reducing bank conflict
    // 4.1 set
    cudaMemset(dC, 0, sizeof(float));

    // 4.2 kernel
    timer.onTimer(3);
    integralReduceBank << <dimGrid, dimBlock >> > (dC, a, b, dx);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    cudaMemcpy(&dIntegral3, dC, sizeof(float), cudaMemcpyDeviceToHost);

    // result
    printf("host : %f\n", hIntegral);
    printf("device ver1 : %f\n", dIntegral1);
    printf("device ver2 : %f\n", dIntegral2);
    printf("device ver3 : %f\n", dIntegral3);

    timer.printTimer();

    // release memory
    cudaFree(dC);

    return 0;
}