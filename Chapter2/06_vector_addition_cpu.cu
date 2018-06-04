#include "stdio.h"
#include<iostream>
//Defining Number of elements in Array
#define N	5
//Defining vector addition function for CPU
void cpuAdd(int *h_a, int *h_b, int *h_c) {
	int tid = 0;	
	while (tid < N)
	{
		h_c[tid] = h_a[tid] + h_b[tid];
		tid += 1;
	}
}

int main(void) {
	int h_a[N], h_b[N], h_c[N];
		//Initializing two arrays for addition
	for (int i = 0; i < N; i++) {
		h_a[i] = 2 * i*i;
		h_b[i] = i;
	}
	//Calling CPU function for vector addition
	cpuAdd (h_a, h_b, h_c);
	//Printing Answer
	printf("Vector addition on CPU\n");
	for (int i = 0; i < N; i++) {
		printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
	}
	return 0;
}
