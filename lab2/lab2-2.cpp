#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include "DS_definitions.h"

double f(double x);

int main(int argc, char** argv)
{
	DS_timer timer(2);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel");

	//** 0. initialize **//
	const double a = strtod(argv[1], NULL);
	const double b = strtod(argv[2], NULL);
	const long n = strtol(argv[3], NULL, 10);

	const double dx = (b - a) / n;

	double serial_area = 0.0;
	double parallel_area = 0.0;
	
	const int numThreads = 16; // thread °³¼ö 
	double parallel_local_area[numThreads];
	memset(parallel_local_area, 0, sizeof(double) * numThreads);


	//** 1. Serial code **//
	timer.onTimer(0);
	//** HERE
	//** Write your code implementing the serial algorithm here
	for (int i = 1; i <= n; ++i) {
		serial_area += (f(a + i * dx) + f(a + (i - 1.0) * dx)) / 2.0 * dx;
	}

	timer.offTimer(0);


	//** 2. Parallel code **//
	timer.onTimer(1);
	//** HERE
	//** Write your code implementing the parallel algorithm here
	#pragma omp parallel num_threads(numThreads)
	{
		#pragma omp for
		for (int i = 1; i <= n; ++i) {
			parallel_local_area[omp_get_thread_num()] += (f(a + i * dx) + f(a + (i - 1.0) * dx)) / 2.0 * dx;
		}
	}

	for (int i = 0; i < numThreads; ++i) {
		parallel_area += parallel_local_area[i];
	}

	timer.offTimer(1);



	//** 3. Result printing code **//
	//** HERE
	//** Write your code that compares results of serial and parallel algorithm
	printf("============== Result ==============\n");
	printf("f(x)=x*x\n");
	printf("range = (%f, %f), n = %d\n", a, b, n);
	printf("[Serial] area = %f\n", serial_area);
	printf("[Parallel] area = %f\n", parallel_area);

	timer.printTimer();
	EXIT_WIHT_KEYPRESS;
}

double f(double x) {
	return x * x;
}