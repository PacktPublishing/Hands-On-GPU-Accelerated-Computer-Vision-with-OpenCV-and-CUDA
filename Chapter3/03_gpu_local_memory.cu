#include <stdio.h>
#define N 5

__global__ void gpu_local_memory(int d_in)
{
	int t_local;    
	t_local = d_in * threadIdx.x;     
	printf("Value of Local variable in current thread is: %d \n", t_local);
}



int main(int argc, char **argv)
{

	printf("Use of Local Memory on GPU:\n");
	gpu_local_memory << <1, N >> >(5);  
	cudaDeviceSynchronize();
	return 0;
}