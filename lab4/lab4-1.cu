#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "DS_timer.h"
#include "DS_definitions.h"

#define NUM_DATA 1024 * 1024 * 128

__global__ void vecAdd(const int *a, const int *b, int* c, int size)
{
    int i = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    // Set timer
    DS_timer timer(5);
    timer.setTimerName(0, (char*)"Host");
    timer.setTimerName(1, (char*)"Device_Total");
    timer.setTimerName(2, (char*)"Device_Computation");
    timer.setTimerName(3, (char*)"Device_DataTransfer(Host->Device)");
    timer.setTimerName(4, (char*)"Device_DataTransfer(Device->Host)");

    // init vectors
    int* a = new int[NUM_DATA];
    int* b = new int[NUM_DATA];
    int* sum_h = new int[NUM_DATA]; // result vector for host

    int* d_a;
    int* d_b;
    int* d_c;
    int* sum_d = new int[NUM_DATA]; // result vector for device
    
    memset(a, 0, sizeof(int) * NUM_DATA);
    memset(b, 0, sizeof(int) * NUM_DATA);
    memset(sum_h, 0, sizeof(int) * NUM_DATA);
    memset(sum_d, 0, sizeof(int) * NUM_DATA);

    for (int i = 0; i < NUM_DATA; ++i) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // host: vector sum
    timer.onTimer(0);
    for (int i = 0; i < NUM_DATA; ++i) {
        sum_h[i] = a[i] + b[i];
    }
    timer.offTimer(0);

    // memory allocation on device
    cudaMalloc(&d_a, sizeof(int) * NUM_DATA);
    cudaMalloc(&d_b, sizeof(int) * NUM_DATA);
    cudaMalloc(&d_c, sizeof(int) * NUM_DATA);

    timer.onTimer(1);

    // memory copy: host -> device
    timer.onTimer(3);
    cudaMemcpy(d_a, a, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);
    timer.offTimer(3);

    int threadsInBlock = 1024;
    dim3 numBlocks(NUM_DATA / threadsInBlock, 1, 1);
    dim3 threadsPerBlock(threadsInBlock, 1, 1);

    // device: vector sum
    timer.onTimer(2);
    vecAdd << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, NUM_DATA);
    cudaDeviceSynchronize();
    timer.offTimer(2);

    // memory copy: device -> host
    timer.onTimer(4);
    cudaMemcpy(sum_d, d_c, sizeof(int) * NUM_DATA, cudaMemcpyDeviceToHost);
    timer.offTimer(4);

    timer.offTimer(1);

    // compare results
    bool result = true;
    for (int i = 0; i < NUM_DATA; ++i) {
        if (sum_h[i] != sum_d[i]) {
            printf("[%d] the results is not matched! (%d.%d)\n", i, sum_h[i], sum_d[i]);

            result = false;
        }
    }

    if (result) {
        printf("GPU works well!\n");
    }

    timer.printTimer();

    // release memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] sum_h;
    delete[] sum_d;

    return 0;
}
