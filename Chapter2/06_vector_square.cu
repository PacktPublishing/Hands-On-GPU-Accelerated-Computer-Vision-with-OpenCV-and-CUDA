#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//Defining number of elements in Array
#define N	5
//Kernel function for squaring number
__global__ void gpuSquare(float *d_in, float *d_out) {
	//Getting thread index for current kernel
	int tid = threadIdx.x;	// handle the data at this index
	float temp = d_in[tid];
	d_out[tid] = temp*temp;
}

int main(void) {
	//Defining Arrays for host
	float h_in[N], h_out[N];
	//Defining Pointers for device
	float *d_in, *d_out;

	// allocate the memory on the cpu
	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));
	//Initializing Array
	for (int i = 0; i < N; i++) {
		h_in[i] = i;
	}
	//Copy Array from host to device
	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	//Calling square kernel with one block and N threads per block
	gpuSquare << <1, N >> >(d_in, d_out);
	//Coping result back to host from device memory
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
	//Printing result on console
	printf("Square of Number on GPU \n");
	for (int i = 0; i < N; i++) {
		printf("The square of %f is %f\n", h_in[i], h_out[i]);
	}
	//Free up memory
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
