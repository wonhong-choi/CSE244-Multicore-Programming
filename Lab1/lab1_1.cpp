#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {

	int numThread = 4;
	#pragma omp parallel num_threads(numThread)
	{
		printf("[Thread %d/%d] Hello OpenMP!\n", omp_get_thread_num(), omp_get_num_threads());
	}

	return 0;
}