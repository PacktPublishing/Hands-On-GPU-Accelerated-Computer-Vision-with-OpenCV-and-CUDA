#include <iostream>

#include <stdio.h>
__global__ void myfirstkernel(void) {
	//blockIdx.x gives the block number of current kernel
	printf("Hello!!!I'm thread in block: %d\n", blockIdx.x);
}

int main(void) {
	//A kernel call with 16 blocks and 1 thread per block
	myfirstkernel << <16,1>> >();
	//Function used for waiting for all kernels to finish
	cudaDeviceSynchronize();
	printf("All threads are finished!\n");
	return 0;
}
