#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "DS_definitions.h"
#include "DS_timer.h"

#define ROW_SIZE 8192
#define COL_SIZE 8192

#define GEN_INT1 rand() % 10 // generate integer which has just 1 digit

// matrix sum (2D grid with 2D block)
__global__ void matAdd1(const int* a, const int* b, int* c, int rowSize, int colSize) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row idx
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col idx

    if (i < rowSize && j < colSize) {
        c[i * colSize + j] = a[i * colSize + j] + b[i * colSize + j];
    }
}

// matrix sum (1D grid with 1D block)
__global__ void matAdd2(const int* a, const int* b, int* c, int rowSize, int colSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // col idx
    
    if (i < rowSize * colSize) {
        c[i] = a[i] + b[i];
    }
}

// matrix sum (2D grid with 1D block)
__global__ void matAdd3(const int* a, const int* b, int* c, int rowSize, int colSize) {
    int i = blockIdx.y; // row idx
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col idx

    if (i < rowSize && j < colSize) {
        c[i * colSize + j] = a[i * colSize + j] + b[i * colSize + j];
    }
}

int main()
{
    // Set timer
    DS_timer timer(6);
    timer.setTimerName(0, (char*)"Host");
    //timer.setTimerName(1, (char*)"Device_Total");
    timer.setTimerName(1, (char*)"Device_Computation_2D2D");
    timer.setTimerName(2, (char*)"Device_Computation_1D1D");
    timer.setTimerName(3, (char*)"Device_Computation_2D1D");
    timer.setTimerName(4, (char*)"Device_DataTransfer(Host->Device)");
    timer.setTimerName(5, (char*)"Device_DataTransfer(Device->Host)");

    // init matrices
    int* a = new int[ROW_SIZE * COL_SIZE];
    int* b = new int[ROW_SIZE * COL_SIZE];
    int* sum_h = new int[ROW_SIZE * COL_SIZE]; // result metrix for host
    
    int* d_a;
    int* d_b;
    int* d_c;

    int* sum_d_2D2D = new int[ROW_SIZE * COL_SIZE]; // result metrix for device
    int* sum_d_1D1D = new int[ROW_SIZE * COL_SIZE]; // result metrix for device
    int* sum_d_2D1D = new int[ROW_SIZE * COL_SIZE]; // result metrix for device
    
    memset(a, 0, sizeof(int) * ROW_SIZE * COL_SIZE);
    memset(b, 0, sizeof(int) * ROW_SIZE * COL_SIZE);
    memset(sum_h, 0, sizeof(int) * ROW_SIZE * COL_SIZE);
    memset(sum_d_2D2D, 0, sizeof(int) * ROW_SIZE * COL_SIZE);
    memset(sum_d_1D1D, 0, sizeof(int) * ROW_SIZE * COL_SIZE);
    memset(sum_d_2D1D, 0, sizeof(int) * ROW_SIZE * COL_SIZE);

    for (int i = 0; i < ROW_SIZE * COL_SIZE; ++i) {
        a[i] = GEN_INT1;
        b[i] = GEN_INT1;
    }

    // host: matrix sum
    timer.onTimer(0);
    for (int i = 0; i < ROW_SIZE; ++i) {
        for (int j = 0; j < COL_SIZE; ++j) {
            sum_h[i * COL_SIZE + j] = a[i * COL_SIZE + j] + b[i * COL_SIZE + j];
        }
    }
    timer.offTimer(0);

    // memory allocation
    cudaMalloc(&d_a, sizeof(int) * ROW_SIZE * COL_SIZE);
    cudaMalloc(&d_b, sizeof(int) * ROW_SIZE * COL_SIZE);
    cudaMalloc(&d_c, sizeof(int) * ROW_SIZE * COL_SIZE);


    // memory copy: host -> device
    timer.onTimer(4);
    cudaMemcpy(d_a, a, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyHostToDevice);
    timer.offTimer(4);
    
    
    // 1. 2D grid with 2D blocks //
    // 1.1. set grid and block
    dim3 dimGrid_2D2D(COL_SIZE / 32, ROW_SIZE / 32, 1);
    dim3 dimBlock_2D2D(32, 32, 1);

    // 1.2. matrix sum by 2D2D
    timer.onTimer(1);
    matAdd1 << <dimGrid_2D2D, dimBlock_2D2D >> > (d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    // 1.3. memory copy: device -> host
    timer.onTimer(5);
    cudaMemcpy(sum_d_2D2D, d_c, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyDeviceToHost);
    timer.offTimer(5);

    // 1.4. compare result (host & 2D2D)
    bool result = true;
    for (int i = 0; i < ROW_SIZE; ++i) {
        for (int j = 0; j < COL_SIZE; ++j) {
            if (sum_h[i * COL_SIZE + j] != sum_d_2D2D[i * COL_SIZE + j]) {
                printf("[%d][%d] the results is not matched! (%d.%d)\n", i, j, sum_h[i * COL_SIZE + j], sum_d_2D2D[i * COL_SIZE + j]);

                result = false;
            }
        }
    }
    if (result) {
        printf("GPU works well! (2D2D)\n");
    }


    // 2. 1D grid with 1D block //
    // 2.1. set grid and block
    dim3 dimGrid_1D1D(ROW_SIZE * COL_SIZE / 1024, 1, 1);
    dim3 dimBlock_1D1D(1024, 1, 1);

    // 2.2. matrix sum by 1D1D
    timer.onTimer(2);
    matAdd2 << <dimGrid_1D1D, dimBlock_1D1D >> > (d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
    cudaDeviceSynchronize();
    timer.offTimer(2);

    // 2.3. memory copy: device -> host
    //timer.onTimer(5);
    cudaMemcpy(sum_d_1D1D, d_c, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyDeviceToHost);
    //timer.offTimer(5);

    // 2.4. compare result (host & 1D1D)
    result = true;
    for (int i = 0; i < ROW_SIZE; ++i) {
        for (int j = 0; j < COL_SIZE; ++j) {
            if (sum_h[i * COL_SIZE + j] != sum_d_1D1D[i * COL_SIZE + j]) {
                printf("[%d][%d] the results is not matched! (%d.%d)\n", i, j, sum_h[i * COL_SIZE + j], sum_d_1D1D[i * COL_SIZE + j]);

                result = false;
            }
        }
    }
    if (result) {
        printf("GPU works well! (1D1D)\n");
    }


    // 3. 2D grid with 1D block //
    // 3.1. set grid and block
    dim3 dimGrid_2D1D(COL_SIZE / 1024, ROW_SIZE, 1);
    dim3 dimBlock_2D1D(1024, 1, 1);

    // 3.2. matrix sum by 2D1D
    timer.onTimer(3);
    matAdd3 << <dimGrid_2D1D, dimBlock_2D1D >> > (d_a, d_b, d_c, ROW_SIZE, COL_SIZE);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    // 3.3. memory copy: device -> host
    //timer.onTimer(5);
    cudaMemcpy(sum_d_2D1D, d_c, sizeof(int) * ROW_SIZE * COL_SIZE, cudaMemcpyDeviceToHost);
    //timer.offTimer(5);

    // 3.4. compare result (host & 2D1D)
    result = true;
    for (int i = 0; i < ROW_SIZE; ++i) {
        for (int j = 0; j < COL_SIZE; ++j) {
            if (sum_h[i * COL_SIZE + j] != sum_d_2D1D[i * COL_SIZE + j]) {
                printf("[%d][%d] the results is not matched! (%d.%d)\n", i, j, sum_h[i * COL_SIZE + j], sum_d_2D1D[i * COL_SIZE + j]);

                result = false;
            }
        }
    }
    if (result) {
        printf("GPU works well! (2D1D)\n");
    }

    
    timer.printTimer();
    
    // release memory
    delete[] a;
    delete[] b;
    delete[] sum_h;
    delete[] sum_d_2D2D;
    delete[] sum_d_1D1D;
    delete[] sum_d_2D1D;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}