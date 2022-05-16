#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "omp.h"
#include "DS_definitions.h"
#include "DS_timer.h"

#define M 1024 // row size of Matrix A
#define K 512 // col size of Matrix A == row size of Matrix B
#define N 1024 // col size of Matrix B

#define GEN_FLOAT ((rand() % 10) + ((rand() / (float)RAND_MAX)))

// matrix multiply (2D grid with 2D block) w/o shared mem
__global__ void matMul(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx
 
    if (i < _m && j < _n) {
        for (int t = 0; t < _k; ++t) {
            c[i * _n + j] += a[i * _k + t] * b[t * _n + j];
            //c[i * _n + j] = __fadd_rn(c[i * _n + j], __fmul_rn(a[i * _k + t], b[t * _n + j]));
        }
    }
}

// matrix multiply (2D grid with 2D block) w shared mem
__global__ void matMulShared(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx
    
    __shared__ float sA[32][32]; // 32 == blockDim.x == blockDim.y
    __shared__ float sB[32][32]; // 32 == blockDim.x == blockDim.y

    int w = _k / 32; // 512/32=16

    if (i < _m && j < _n) {
        for (int t = 0; t < w; ++t) {
            sA[threadIdx.y][threadIdx.x] = a[i * _k + t * 32 + threadIdx.x];
            sB[threadIdx.y][threadIdx.x] = b[(t * 32 + threadIdx.y) * _n + j];
            __syncthreads();


            for (int p = 0; p < 32; ++p) {
                c[i * _n + j] += sA[threadIdx.y][p] * sB[p][threadIdx.x];
            }
            __syncthreads();
        }
    }
}

// matrix multiply (2D grid with 2D block) w register
__global__ void matMulReg(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx

    __shared__ float sA[32][32]; // 32 == blockDim.x == blockDim.y
    __shared__ float sB[32][32]; // 32 == blockDim.x == blockDim.y

    float value = 0.0;

    int w = _k / 32; // 512/32=16

    if (i < _m && j < _n) {
        for (int t = 0; t < w; ++t) {
            sA[threadIdx.y][threadIdx.x] = a[i * _k + t * 32 + threadIdx.x];
            sB[threadIdx.y][threadIdx.x] = b[(t * 32 + threadIdx.y) * _n + j];
            __syncthreads();


            for (int p = 0; p < 32; ++p) {
                value += sA[threadIdx.y][p] * sB[p][threadIdx.x];
            }
            __syncthreads();
        }
        c[i * _n + j] = value;
    }
}



int main()
{
    // Set timer
    DS_timer timer(6);
    timer.setTimerName(0, (char*)"Host_Serial");
    timer.setTimerName(1, (char*)"Device_kernel w/o shared");
    timer.setTimerName(2, (char*)"Device_kernel w shared");
    timer.setTimerName(3, (char*)"Device_kernel w reg");
    timer.setTimerName(4, (char*)"Device_DataTransfer(Host->Device)");
    timer.setTimerName(5, (char*)"Device_DataTransfer(Device->Host)");

    // init matrices
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* hC = new float[M * N]; // result matrix for host (serial)
    
    float* d_a;
    float* d_b;
    float* d_c;

    float* dC1 = new float[M * N]; // result matrix for device (w/o shared mem)
    float* dC2 = new float[M * N]; // result matrix for device (w shared mem)
    float* dC3 = new float[M * N]; // result matrix for device (w register)
    
    memset(a, 0, sizeof(float) * M * K);
    memset(b, 0, sizeof(float) * K * N);
    memset(hC, 0, sizeof(float) * M * N);
    memset(dC1, 0, sizeof(float) * M * N);
    memset(dC2, 0, sizeof(float) * M * N);
    memset(dC3, 0, sizeof(float) * M * N);
    
    for (int i = 0; i < M * K; ++i) {
        a[i] = GEN_FLOAT;
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = GEN_FLOAT;
    }

    // 1. Host serial: matrix multiplication
    timer.onTimer(0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                hC[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
    timer.offTimer(0);

    
    // 2. Device w/o shared mem
    // 2.0. initailize memory
    
    // memory allocation
    cudaMalloc(&d_a, sizeof(float) * M * K);
    cudaMalloc(&d_b, sizeof(float) * K * N);
    cudaMalloc(&d_c, sizeof(float) * M * N);
    
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // memory copy: host -> device
    timer.onTimer(4);
    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    timer.offTimer(4);

    // 2.1. set grid and block
    int threadSize = 32;
    dim3 dimBlock(threadSize, threadSize, 1);
    dim3 dimGrid(int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1);

    // 2.2. matrix mul
    timer.onTimer(1);
    matMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    // 2.3. memory copy: device -> host
    timer.onTimer(5);
    cudaMemcpy(dC1, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    timer.offTimer(5);


    // 3. Device w shared mem
    // 3.0. initailize memory
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // 3.1. set grid and block
    /*int threadSize = 32;
    dim3 dimBlock(threadSize, threadSize, 1);
    dim3 dimGrid(int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1);*/

    // 3.2. matrix mul
    timer.onTimer(2);
    matMulShared << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(2);

    // 3.3. memory copy: device -> host
    cudaMemcpy(dC2, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 4. Device w register
    // 4.0. initailize memory
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // 4.1. set grid and block
    /*int threadSize = 32;
    dim3 dimBlock(threadSize, threadSize, 1);
    dim3 dimGrid(int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1);*/

    // 4.2. matrix mul
    timer.onTimer(3);
    matMulReg << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    // 4.3. memory copy: device -> host
    cudaMemcpy(dC3, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 5. report result 
    // 5.0. info 
    printf("Size: A=(%d x %d), B=(%d x %d), C=(%d x %d)\n", M, K, K, N, M, N);
    printf("Grid(%d,%d,%d), Block(%d,%d,%d)\n\n", int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1, threadSize, threadSize, 1);
    
    // 5.1. compare result (Host serial & Device w/o shared mem)
    float mae = 0.0; // mean absolute error
    int errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC1[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, mul_h1[i * N + j], mul_d[i * N + j]);
                mae += abs(hC[i * N + j] - dC1[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt==0) {
        printf("GPU w/o shared mem works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA w/o shared mem) : %f\n", mae / errCnt);
    }

    // 4.2. compare result (Host serial & Device w shared mem)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC2[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, mul_h1[i * N + j], mul_d[i * N + j]);
                mae += abs(hC[i * N + j] - dC2[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("GPU w shared mem works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA w shared mem) : %f\n", mae / errCnt);
    }

    // 4.3. compare result (Host serial & Device w register)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC3[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, mul_h1[i * N + j], mul_d[i * N + j]);
                mae += abs(hC[i * N + j] - dC3[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("GPU w shared mem works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA w register) : %f\n", mae / errCnt);
    }


    timer.printTimer();
    
    // release memory
    delete[] a;
    delete[] b;
    delete[] hC;
    delete[] dC1;
    delete[] dC2;
    delete[] dC3;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}