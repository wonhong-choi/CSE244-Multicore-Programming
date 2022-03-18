#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

// Set the size of matrix and vector
// matrix A = m by n
// vector b = n by 1
#define m (10000)
#define n (10000)

#define GenFloat (rand() % 100 + ((float)(rand() % 100) / 100.0))
void genRandomInput();

float A[m][n];
float X[n];
float Y_serial[m];
float Y_parallel[m];

int main(int argc, char** argv)
{
	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");

	genRandomInput();

	int numThreads = atoi(argv[1]);

	//** 1. Serial code **//
	timer.onTimer(0);


	//** HERE
	//** Write your code implementing the serial algorithm here
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			Y_serial[i] += A[i][j] * X[j];
		}
	}

	timer.offTimer(0);



	//** 2. Parallel code **//
	timer.onTimer(1);


	//** HERE
	//** Write your code implementing the parallel algorithm here
	#pragma omp parallel num_threads(numThreads)
	{
		#pragma omp for
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				Y_parallel[i] += A[i][j] * X[j];
			}
		}
	}
	

	timer.offTimer(1);



	//** 3. Result checking code **//
	bool isCorrect = true;

	//** HERE
	//** Wriet your code that compares results of serial and parallel algorithm
	// Set the flag 'isCorrect' to true when they are matched
	for (int i = 0; i < m; ++i) {
		if (Y_serial[i] != Y_parallel[i]) {
			isCorrect = false;
			break;
		}
	}



	if (isCorrect)
		printf("Results are matched :)\n");
	else
		printf("Results are not matched! :(\n");

	timer.printTimer();
	EXIT_WIHT_KEYPRESS;
}

void genRandomInput(void) {
	// A matrix
	LOOP_INDEX(row, m) {
		LOOP_INDEX(col, n) {
			A[row][col] = GenFloat;
		}
	}

	LOOP_I(n)
		X[i] = GenFloat;

	memset(Y_serial, 0, sizeof(float) * m);
	memset(Y_parallel, 0, sizeof(float) * m);
}
