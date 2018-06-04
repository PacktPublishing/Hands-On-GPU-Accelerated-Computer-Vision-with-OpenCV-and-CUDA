
#include <stdio.h>

__global__ void gpu_shared_memory(float *d_a)
{
	// Defining local variables which are private to each thread
	int i, index = threadIdx.x;
	float average, sum = 0.0f;
	//Define shared memory
	__shared__ float sh_arr[10];

	
	sh_arr[index] = d_a[index];

	__syncthreads();    // This ensures all the writes to shared memory have completed

	for (i = 0; i<= index; i++) 
	{ 
		sum += sh_arr[i]; 
	}
	average = sum / (index + 1.0f);

	d_a[index] = average; 

	sh_arr[index] = average;
}

int main(int argc, char **argv)
{
	//Define Host Array
	float h_a[10];   
	//Define Device Pointer
	float *d_a;       
	
	for (int i = 0; i < 10; i++) {
		h_a[i] = i;
	}
	// allocate global memory on the device
	cudaMalloc((void **)&d_a, sizeof(float) * 10);
	// now copy data from host memory  to device memory 
	cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
	
	gpu_shared_memory << <1, 10 >> >(d_a);
	// copy the modified array back to the host memory
	cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	printf("Use of Shared Memory on GPU:  \n");
	//Printing result on console
	for (int i = 0; i < 10; i++) {
		printf("The running average after %d element is %f \n", i, h_a[i]);
	}
	return 0;
}
