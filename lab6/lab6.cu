#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "omp.h"
#include "DS_definitions.h"
#include "DS_timer.h"

#define M 1024 // row size of Matrix A
#define K 2048 // col size of Matrix A == row size of Matrix B
#define N 1024 // col size of Matrix B

#define BLOCK_SIZE 16
#define TILE_SIZE 16   // TILE_SIZE == BLOCK_SIZE

#define GEN_FLOAT ((rand() % 10) + ((rand() / (float)RAND_MAX)))

// matrix multiply (2D grid with 2D block) w/o shared mem
__global__ void matMul(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx
 
    if (i < _m && j < _n) {
        for (int t = 0; t < _k; ++t) {
            c[i * _n + j] = __fadd_rn(c[i * _n + j], __fmul_rn(a[i * _k + t], b[t * _n + j]));
        }
    }
}


// matrix multiply (2D grid with 2D block) w shared mem
__global__ void matMulShared(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx
    
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < ceil((float)_k / TILE_SIZE); ++t) {
        if (i >= _m || t * TILE_SIZE + threadIdx.x >= _k) {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        else {
            sA[threadIdx.y][threadIdx.x] = a[i * _k + t * TILE_SIZE + threadIdx.x];
        }
        
        if (j >= _n || t * TILE_SIZE + threadIdx.y >= _k) {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        else {
            sB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * _n + j];
        }
        __syncthreads();


        if (i < _m && j < _n) {
            for (int p = 0; p < TILE_SIZE; ++p) {
                c[i * _n + j] = __fadd_rn(c[i * _n + j], __fmul_rn(sA[threadIdx.y][p], sB[p][threadIdx.x]));
            }
        }
        __syncthreads();
    }
}


// matrix multiply (2D grid with 2D block) w register
__global__ void matMulReg(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row idx
    int j = blockDim.x * blockIdx.x + threadIdx.x; // col idx

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float value = 0.0;

    for (int t = 0; t < ceil((float)_k / TILE_SIZE); ++t) {
        if (i >= _m || t * TILE_SIZE + threadIdx.x >= _k) {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        else {
            sA[threadIdx.y][threadIdx.x] = a[i * _k + t * TILE_SIZE + threadIdx.x];
        }

        if (j >= _n || t * TILE_SIZE + threadIdx.y >= _k) {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        else {
            sB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * _n + j];
        }
        __syncthreads();

        for (int p = 0; p < TILE_SIZE; ++p) {
            value = __fadd_rn(value, __fmul_rn(sA[threadIdx.y][p], sB[p][threadIdx.x]));
        }
        __syncthreads();
    }
    if (i >= _m || j >= _n)
        return;
    c[i * _n + j] = value;
}


// matrix multiply (2D grid with 2D block) w bank conflict
__global__ void matMulWBank(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int j = blockDim.y * blockIdx.y + threadIdx.y; // col idx   // transposed
    int i = blockDim.x * blockIdx.x + threadIdx.x; // row idx

    __shared__ float sA[TILE_SIZE][TILE_SIZE]; 
    __shared__ float sB[TILE_SIZE][TILE_SIZE]; 

    float value = 0.0;

    for (int t = 0; t < ceil((float)_k / TILE_SIZE); ++t) {
        if (i >= _m || t * TILE_SIZE + threadIdx.y >= _k) {
            sA[threadIdx.x][threadIdx.y] = 0.0;
        }
        else {
            sA[threadIdx.x][threadIdx.y] = a[i * _k + t * TILE_SIZE + threadIdx.y];
        }

        if (j >= _n || t * TILE_SIZE + threadIdx.x >= _k) {
            sB[threadIdx.x][threadIdx.y] = 0.0;
        }
        else {
            sB[threadIdx.x][threadIdx.y] = b[(t * TILE_SIZE + threadIdx.x) * _n + j];
        }
        __syncthreads();

        for (int p = 0; p < TILE_SIZE; ++p) {
            value = __fadd_rn(value, __fmul_rn(sA[threadIdx.x][p], sB[p][threadIdx.y]));
        }
        __syncthreads();
    }
    if (i >= _m || j >= _n)
        return;
    c[i * _n + j] = value;
}

// matrix multiply (2D grid with 2D block) w NO bank conflict
__global__ void matMulNoBank(const float* a, const float* b, float* c, int _m, int _k, int _n) {
    int j = blockDim.y * blockIdx.y + threadIdx.y; // col idx   // transposed
    int i = blockDim.x * blockIdx.x + threadIdx.x; // row idx

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float value = 0.0;

    for (int t = 0; t < ceil((float)_k / TILE_SIZE); ++t) {
        if (i >= _m || t * TILE_SIZE + threadIdx.y >= _k) {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        else {
            sA[threadIdx.y][threadIdx.x] = a[i * _k + t * TILE_SIZE + threadIdx.y];
        }

        if (j >= _n || t * TILE_SIZE + threadIdx.x >= _k) {
            sB[threadIdx.x][threadIdx.y] = 0.0;
        }
        else {
            sB[threadIdx.x][threadIdx.y] = b[(t * TILE_SIZE + threadIdx.x) * _n + j];
        }
        __syncthreads();

        for (int p = 0; p < TILE_SIZE; ++p) {
            value = __fadd_rn(value, __fmul_rn(sA[p][threadIdx.x], sB[p][threadIdx.y]));
        }
        __syncthreads();
    }
    if (i >= _m || j >= _n)
        return;
    c[i * _n + j] = value;
}





int main()
{
    // Set timer
    DS_timer timer(8);
    timer.setTimerName(0, (char*)"Host_Serial");
    timer.setTimerName(1, (char*)"Device_kernel w/o shared");
    timer.setTimerName(2, (char*)"Device_kernel w shared");
    timer.setTimerName(3, (char*)"Device_kernel w reg");
    timer.setTimerName(4, (char*)"Device_kernel w bank conflict");
    timer.setTimerName(5, (char*)"Device_kernel w No bank conflict");
    timer.setTimerName(6, (char*)"Device_DataTransfer(Host->Device)");
    timer.setTimerName(7, (char*)"Device_DataTransfer(Device->Host)");

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
    float* dC4 = new float[M * N]; // result matrix for device (w bank conflict)
    float* dC5 = new float[M * N]; // result matrix for device (w no bank conflict)
    
    memset(a, 0, sizeof(float) * M * K);
    memset(b, 0, sizeof(float) * K * N);
    memset(hC, 0, sizeof(float) * M * N);
    memset(dC1, 0, sizeof(float) * M * N);
    memset(dC2, 0, sizeof(float) * M * N);
    memset(dC3, 0, sizeof(float) * M * N);
    memset(dC4, 0, sizeof(float) * M * N);
    memset(dC5, 0, sizeof(float) * M * N);
    
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
    timer.onTimer(6);
    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    timer.offTimer(6);

    // 2.1. set grid and block
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(int(ceil(N / (float)BLOCK_SIZE)), int(ceil(M / (float)BLOCK_SIZE)), 1);

    // 2.2. matrix mul
    timer.onTimer(1);
    matMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    // 2.3. memory copy: device -> host
    timer.onTimer(7);
    cudaMemcpy(dC1, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    timer.offTimer(7);


    // 3. Device w shared mem
    // 3.0. initailize memory
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // 3.1. set grid and block

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

    // 4.2. matrix mul
    timer.onTimer(3);
    matMulReg << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    // 4.3. memory copy: device -> host
    cudaMemcpy(dC3, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 5. Device w bank conflict
    // 5.0. initailize memory
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // 5.1. set grid and block
    
    // 5.2. matrix mul
    timer.onTimer(4);
    matMulWBank << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(4);

    // 5.3. memory copy: device -> host
    cudaMemcpy(dC4, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    
    // 6. Device w bank conflict
    // 6.0. initailize memory
    cudaMemset(d_c, 0, sizeof(float) * M * N); // init d_c

    // 6.1. set grid and block

    // 6.2. matrix mul
    timer.onTimer(5);
    matMulNoBank << <dimGrid, dimBlock >> > (d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.offTimer(5);

    // 6.3. memory copy: device -> host
    cudaMemcpy(dC5, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    // 7. report result 
    // 7.0. info 
    printf("Size: A=(%d x %d), B=(%d x %d), C=(%d x %d)\n", M, K, K, N, M, N);
    printf("Grid(%d,%d,%d), Block(%d,%d,%d)\n\n", int(ceil(N / (float)BLOCK_SIZE)), int(ceil(M / (float)BLOCK_SIZE)), 1, BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // 6.1. compare result (Host serial & Device w/o shared mem)
    float mae = 0.0; // mean absolute error
    int errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC1[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, hC[i * N + j], dC1[i * N + j]);
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

    // 7.2. compare result (Host serial & Device w shared mem)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC2[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, hC[i * N + j], dC2[i * N + j]);
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

    // 7.3. compare result (Host serial & Device w register)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC3[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, hC[i * N + j], dC3[i * N + j]);
                mae += abs(hC[i * N + j] - dC3[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("GPU w reg works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA w register) : %f\n", mae / errCnt);
    }

    // 7.4. compare result (Host serial & Device W bank conflict)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC4[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, hC[i * N + j], dC4[i * N + j]);
                mae += abs(hC[i * N + j] - dC4[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("GPU W Bank Conflict works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA W Bank Conflict) : %f\n", mae / errCnt);
    }

    // 7.5. compare result (Host serial & Device W NO bank conflict)
    mae = 0.0; // mean absolute error
    errCnt = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (hC[i * N + j] != dC5[i * N + j]) {
                //printf("Not matched! C[%d][%d]: Ground-Truth %f, device %f\n", i, j, hC[i * N + j], dC5[i * N + j]);
                mae += abs(hC[i * N + j] - dC5[i * N + j]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("GPU W No bank conflict works well!\n");
    }
    else {
        printf("Mean absolute Error(serial & CUDA W No bank conflict) : %f\n", mae / errCnt);
    }


    timer.printTimer();
    
    // release memory
    delete[] a;
    delete[] b;
    delete[] hC;
    delete[] dC1;
    delete[] dC2;
    delete[] dC3;
    delete[] dC4;
    delete[] dC5;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}