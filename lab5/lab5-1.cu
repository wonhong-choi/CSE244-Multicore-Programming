#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "omp.h"
#include "DS_definitions.h"
#include "DS_timer.h"

#define NUM_THREADS 6 // the number of threads (OpenMP)

#define M 972 // row size of Matrix A
#define K 2048 // col size of Matrix A == row size of Matrix B
#define N 1024 // col size of Matrix B

#define GEN_FLOAT ((rand() % 10) + ((rand() / (float)RAND_MAX)))

// matrix multiply (2D grid with 2D block)
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


int main()
{
    // Set timer
    DS_timer timer(6);
    timer.setTimerName(0, (char*)"Host_Serial");
    timer.setTimerName(1, (char*)"Host_OpenMP");
    timer.setTimerName(2, (char*)"Device_Total");
    timer.setTimerName(3, (char*)"Device_Computation(kernel)");
    timer.setTimerName(4, (char*)"Device_DataTransfer(Host->Device)");
    timer.setTimerName(5, (char*)"Device_DataTransfer(Device->Host)");

    // init matrices
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* mul_h1 = new float[M * N]; // result matrix for host (serial)
    float* mul_h2 = new float[M * N]; // result matrix for host (using OpenMP)
    
    float* d_a;
    float* d_b;
    float* d_c;

    float* mul_d = new float[M * N]; // result matrix for device
    
    memset(a, 0, sizeof(float) * M * K);
    memset(b, 0, sizeof(float) * K * N);
    memset(mul_h1, 0, sizeof(float) * M * N);
    memset(mul_h2, 0, sizeof(float) * M * N);
    memset(mul_d, 0, sizeof(float) * M * N);
    
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
                mul_h1[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
    timer.offTimer(0);

    // 2. Host OpenMP: matrix multiplication
    timer.onTimer(1);
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                mul_h2[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
    timer.offTimer(1);

    // memory allocation
    cudaMalloc(&d_a, sizeof(float) * M * K);
    cudaMalloc(&d_b, sizeof(float) * K * N);
    cudaMalloc(&d_c, sizeof(float) * M * N);
    
    timer.onTimer(2);

    // memory copy: host -> device
    timer.onTimer(4);
    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    timer.offTimer(4);
    
    cudaMemset(d_c, 0, sizeof(float) * M * N);

    
    // 3. Device: 2D grid with 2D blocks
    // 3.1. set grid and block
    int threadSize = 32;
    dim3 dimBlock(threadSize, threadSize, 1);
    dim3 dimGrid(int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1);

    // 3.2. matrix mul
    timer.onTimer(3);
    matMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    // 3.3. memory copy: device -> host
    timer.onTimer(5);
    cudaMemcpy(mul_d, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    timer.offTimer(5);

    timer.offTimer(2);


    // 4. report result 
    // 4.0. info 
    printf("Size: A=(%d x %d), B=(%d x %d), C=(%d x %d)\n", M, K, K, N, M, N);
    printf("Grid(%d,%d,%d), Block(%d,%d,%d)\n\n", int(ceil(N / (float)threadSize)), int(ceil(M / (float)threadSize)), 1, threadSize, threadSize, 1);
    
    // 4.1. compare result (Host serial & Host OpenMP)
    bool result = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (mul_h1[i * N + j] != mul_h2[i * N + j]) {
                //printf("[%d][%d] the results is not matched! (%f.%f)\n", i, j, mul_h1[i * N + j], mul_h2[i * N + j]);

                result = false;
            }
        }
    }
    if (result) {
        printf("OpenMP [%d] works well!\n", NUM_THREADS);
    }

    // 4.2. compare result (Host serial & Device)
    float mae = 0.0; // mean absolute error
    int errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (mul_h1[i * N + j] != mul_d[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, mul_h1[i * N + j], mul_d[i * N + j]);
                mae += abs(mul_h1[i * N + j] - mul_d[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt==0) {
        printf("GPU works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA) : %f\n", mae / errCnt);
    }
    

    timer.printTimer();
    
    // release memory
    delete[] a;
    delete[] b;
    delete[] mul_h1;
    delete[] mul_h2;
    delete[] mul_d;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}