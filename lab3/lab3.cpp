#include <vector>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "DS_timer.h"
#include "DS_definitions.h"

using namespace std;

#define GenFloat (rand() % 10 + ((float)(rand() % 100) / 100.0))


int main() {
        DS_timer timer(4);
        timer.setTimerName(0, (char*)"Serial");
        timer.setTimerName(1, (char*)"Parallel_ver1");
        timer.setTimerName(2, (char*)"Parallel_ver2");
        timer.setTimerName(3, (char*)"Parallel_ver3");

        srand(4);

        //** 0. initialize **//
        const int SIZE = 1024 * 1024 * 1024;
        double* datas = new double[SIZE];
        for (int i = 0; i < SIZE; ++i) {
            datas[i] = GenFloat;
        }

        const int BIN_SIZE = 10;
        int* serialBin = new int[BIN_SIZE];
        int* parallelBin1 = new int[BIN_SIZE];
        int* parallelBin2 = new int[BIN_SIZE];
        int* parallelBin3 = new int[BIN_SIZE];
        memset(serialBin, 0, sizeof(int) * BIN_SIZE);
        memset(parallelBin1, 0, sizeof(int) * BIN_SIZE);
        memset(parallelBin2, 0, sizeof(int) * BIN_SIZE);
        memset(parallelBin3, 0, sizeof(int) * BIN_SIZE);

    
        const int numThreads = 8;
    
    
    //** 1. Serial code **//
    
    timer.onTimer(0);   // start
    for (int i = 0; i < SIZE; ++i) {
        serialBin[(int)(datas[i])] += 1;
    }
    timer.offTimer(0);  // end


    
    //** 2. Parallel ver1 code **//

    timer.onTimer(1);   // start
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < SIZE; ++i) {
            #pragma omp atomic
                parallelBin1[(int)(datas[i])] += 1;
        }
    }
    timer.offTimer(1);   // end

    
    //** 3. Parallel ver2 code **//
       
    // init local bin
    int** localBins = new int* [numThreads];
    for (int i = 0; i < numThreads; ++i) {
        localBins[i] = new int[BIN_SIZE];
        memset(localBins[i], 0, sizeof(int) * BIN_SIZE);
    }

    //timer.onTimer(2);   // start
    //#pragma omp parallel num_threads(numThreads)
    //{
    //    int tID = omp_get_thread_num();
    //    #pragma omp for
    //    for (int i = 0; i < SIZE; ++i) {
    //        localBins[tID][(int)(datas[i])] += 1;
    //    }
    //    // there is an implicit barrier here
    //    for (int i = 0; i < BIN_SIZE; ++i) {
    //        #pragma omp critical
    //        {
    //            parallelBin2[i] += localBins[tID][i];
    //        }
    //    }
    //}
    //timer.offTimer(2);  // end

    timer.onTimer(2);   // start
    #pragma omp parallel num_threads(numThreads)
    {
        int tID = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < SIZE; ++i) {
            localBins[tID][(int)(datas[i])] += 1;
        }
        // there is an implicit barrier here
        for (int i = 0; i < BIN_SIZE; ++i) {
            #pragma omp atomic
                parallelBin2[i] += localBins[tID][i];
        }
    }
    timer.offTimer(2);  // end


    //** 4. Parallel ver3 code **//
    
    // init local bin
    int** localBins2 = new int* [numThreads];
    for (int i = 0; i < numThreads; ++i) {
        localBins2[i] = new int[BIN_SIZE];
        memset(localBins2[i], 0, sizeof(int) * BIN_SIZE);
    }

    timer.onTimer(3);   // start
    #pragma omp parallel num_threads(numThreads)
    {
        int tID = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < SIZE; ++i) {
            localBins2[tID][(int)(datas[i])] += 1;
        }
        // there is an implicit barrier here
        #pragma omp for
        for (int i = 0; i < BIN_SIZE; ++i) {
            for (int j = 0; j < numThreads; ++j) {
                parallelBin3[i] += localBins2[j][i];
            }
        }
    }
    timer.offTimer(3);  // end

    
    //** 5. Result checking code **//

    // print results
    printf("idx\t\tS\t\tv1\t\tv2\t\tv3\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d\t%d\t%d\t%d\t%d\n", i,serialBin[i], parallelBin1[i], parallelBin2[i], parallelBin3[i]);
    }
    printf("\n");

    // 1) check whether serial[i] is equal to ver1[i] or not
    bool isCorrect = true;
    for (int i = 0; i < BIN_SIZE; ++i) {
        if (serialBin[i] != parallelBin1[i]) {
            printf("serial and parallel Bin1 are different!\n");
            isCorrect = false;
            break;
        }
    }
    if (isCorrect) {
        printf("serial and parallel Bin1 are same!\n");
    }

    // 2) check whether serial[i] is equal to ver2[i] or not
    isCorrect = true;
    for (int i = 0; i < BIN_SIZE; ++i) {
        if (serialBin[i] != parallelBin2[i]) {
            printf("serial and parallel Bin2 are different!\n");
            isCorrect = false;
            break;
        }
    }
    if (isCorrect) {
        printf("serial and parallel Bin2 are same!\n");
    }

    // 3) check whether serial[i] is equal to ver3[i] or not
    isCorrect = true;
    for (int i = 0; i < BIN_SIZE; ++i) {
        if (serialBin[i] != parallelBin3[i]) {
            printf("serial and parallel Bin3 are different!\n");
            isCorrect = false;
            break;
        }
    }
    if (isCorrect) {
        printf("serial and parallel Bin3 are same!\n");
    }

    timer.printTimer();


    // release dynamic array
    delete[] datas;
    for (int i = 0; i < numThreads; ++i) {
        delete[] localBins[i];
    }
    delete[] localBins;
    for (int i = 0; i < numThreads; ++i) {
        delete[] localBins2[i];
    }
    delete[] localBins2;
    
    delete[] serialBin;
    delete[] parallelBin1;
    delete[] parallelBin2;
    delete[] parallelBin3;

    EXIT_WIHT_KEYPRESS;
}