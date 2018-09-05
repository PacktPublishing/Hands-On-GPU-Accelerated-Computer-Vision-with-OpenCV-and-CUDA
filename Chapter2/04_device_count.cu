#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// Main Program 

int main(void)
{
	int device_Count = 0;
	cudaGetDeviceCount(&device_Count);
	// This function returns count of number of CUDA enable devices and 0 if there are no CUDA capable devices.
	if (device_Count == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", device_Count);
	}

	
}
