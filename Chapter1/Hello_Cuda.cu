#include <iostream>

#include <stdio.h>
__global__ void myfirstkernel(void) {
}

int main(void) {
	myfirstkernel << <1, 1 >> >();
	printf("Hello, CUDA!\n");
	return 0;
}
