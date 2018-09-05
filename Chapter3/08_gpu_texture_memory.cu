#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#define NUM_THREADS 10
#define N 10
texture <float, 1, cudaReadModeElementType> textureRef;
__global__ void gpu_texture_memory(int n, float *d_out)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < n) {
		float temp = tex1D(textureRef, float(idx));
		d_out[idx] = temp;
	}
}

int main()
{
	//Calculate number of blocks to launch
	int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
	//Declare device pointer
	float *d_out;
	// allocate space on the device for the result
	cudaMalloc((void**)&d_out, sizeof(float) * N);
	// allocate space on the host for the results
	float *h_out = (float*)malloc(sizeof(float)*N);
	//Declare and initialize host array
	float h_in[N];
	for (int i = 0; i < N; i++) {
		h_in[i] = float(i);
	}
	//Define CUDA Array
	cudaArray *cu_Array;
	cudaMallocArray(&cu_Array, &textureRef.channelDesc, N, 1);
	//Copy data to CUDA Array
	cudaMemcpyToArray(cu_Array, 0, 0, h_in, sizeof(float)*N, cudaMemcpyHostToDevice);
	
	// bind a texture to the CUDA array
	cudaBindTextureToArray(textureRef, cu_Array);
	//Call Kernel	
  	gpu_texture_memory << <num_blocks, NUM_THREADS >> >(N, d_out);
	
	// copy result back to host
	cudaMemcpy(h_out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);
	printf("Use of Texture memory on GPU: \n");
	for (int i = 0; i < N; i++) {
		printf("Texture element at %d is : %f\n",i, h_out[i]);
	}
	free(h_out);
	cudaFree(d_out);
	cudaFreeArray(cu_Array);
	cudaUnbindTexture(textureRef);
	
}
