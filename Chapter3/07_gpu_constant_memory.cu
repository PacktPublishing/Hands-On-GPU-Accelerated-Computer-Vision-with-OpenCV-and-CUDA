#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//Defining two constants
__constant__ int constant_f;
__constant__ int constant_g;
#define N	5
//Kernel function for using constant memory
__global__ void gpu_constant_memory(float *d_in, float *d_out) {
	//Thread index for current kernel
	int tid = threadIdx.x;	
	d_out[tid] = constant_f*d_in[tid] + constant_g;
}

int main(void) {
	//Defining Arrays for host
	float h_in[N], h_out[N];
	//Defining Pointers for device
	float *d_in, *d_out;
	int h_f = 2;
	int h_g = 20;
	// allocate the memory on the cpu
	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));
	//Initializing Array
	for (int i = 0; i < N; i++) {
		h_in[i] = i;
	}
	//Copy Array from host to device
	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	//Copy constants to constant memory
	cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int));

	//Calling kernel with one block and N threads per block
	gpu_constant_memory << <1, N >> >(d_in, d_out);
	//Coping result back to host from device memory
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
	//Printing result on console
	printf("Use of Constant memory on GPU \n");
	for (int i = 0; i < N; i++) {
		printf("The expression for input %f is %f\n", h_in[i], h_out[i]);
	}
	//Free up memory
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
