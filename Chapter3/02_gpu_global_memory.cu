#include <stdio.h>
#define N 5

__global__ void gpu_global_memory(int *d_a)
{
	// "array" is a pointer into global memory on the device
	d_a[threadIdx.x] = threadIdx.x;
}



int main(int argc, char **argv)
{
	// Define Host Array
	int h_a[N];
	//Define device pointer	
	int *d_a;       

						
	cudaMalloc((void **)&d_a, sizeof(int) *N);
	// now copy data from host memory to device memory 
	cudaMemcpy((void *)d_a, (void *)h_a, sizeof(int) *N, cudaMemcpyHostToDevice);
	// launch the kernel 
	gpu_global_memory << <1, N >> >(d_a);  
	// copy the modified array back to the host memory
	cudaMemcpy((void *)h_a, (void *)d_a, sizeof(int) *N, cudaMemcpyDeviceToHost);
	printf("Array in Global Memory is: \n");
	//Printing result on console
	for (int i = 0; i < N; i++) {
		printf("At Index: %d --> %d \n", i, h_a[i]);
	}

	
	return 0;
}