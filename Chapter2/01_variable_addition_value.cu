#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

//Definition of kernel function to add two variables
__global__ void gpuAdd(int d_a, int d_b, int *d_c) {
	*d_c = d_a + d_b;
}

//main function
int main(void) {
	//Defining host variable to store answer
	int h_c;
	//Defining device pointer
	int *d_c;
	//Allocating memory for device pointer
	cudaMalloc((void**)&d_c, sizeof(int));
	//Kernel call by passing 1 and 4 as inputs and storing answer in d_c
	//<< <1,1> >> means 1 block is executed with 1 thread per block
	gpuAdd << <1, 1 >> > (1, 4, d_c);
	//Copy result from device memory to host memory
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("1 + 4 = %d\n", h_c);
	//Free up memory
	cudaFree(d_c);
	return 0;
}
