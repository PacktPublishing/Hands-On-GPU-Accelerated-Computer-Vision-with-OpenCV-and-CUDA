#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
//Kernel function to add two variables, parameters are passed by reference
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
	*d_c = *d_a + *d_b;
}

int main(void) {
	//Defining host variables
	int h_a,h_b, h_c;
	//Defining Device Pointers
	int *d_a,*d_b,*d_c;
	//Initializing host variables
	h_a = 1;
	h_b = 4;
	//Allocating memory for Device Pointers
	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));
	//Coping value of host variables in device memory
	cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
	//Calling kernel with one thread and one block with parameters passed by reference
	gpuAdd << <1, 1 >> > (d_a, d_b, d_c);
	//Coping result from device memory to host
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Passing Parameter by Reference Output: %d + %d = %d\n", h_a, h_b, h_c);
	//Free up memory 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
