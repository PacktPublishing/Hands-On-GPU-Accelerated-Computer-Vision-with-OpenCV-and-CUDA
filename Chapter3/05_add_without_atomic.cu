#include <stdio.h>


#define NUM_THREADS 10000
#define SIZE  10

#define BLOCK_WIDTH 100

__global__ void gpu_increment_without_atomic(int *d_a)
{
	// Calculate thread id for current thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread increments elements wrapping at SIZE variable
	tid = tid % SIZE;
	d_a[tid] += 1;
}

int main(int argc, char **argv)
{

	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

	// declare and allocate host memory
	int h_a[SIZE];
	const int ARRAY_BYTES = SIZE * sizeof(int);

	// declare and allocate GPU memory
	int * d_a;
	cudaMalloc((void **)&d_a, ARRAY_BYTES);
	//Initialize GPU memory to zero
	cudaMemset((void *)d_a, 0, ARRAY_BYTES);

	gpu_increment_without_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);


	// copy back the array to host memory
	cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);


	printf("Number of times a particular Array index has been incremented without atomic add is: \n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("index: %d --> %d times\n ", i, h_a[i]);
	}

	cudaFree(d_a);
	return 0;
}