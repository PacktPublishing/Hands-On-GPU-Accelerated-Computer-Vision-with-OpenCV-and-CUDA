#include <stdio.h>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
	printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int device_Count = 0;
	cudaGetDeviceCount(&device_Count);

	// This function call returns 0 if there are no CUDA capable devices.
	if (device_Count == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", device_Count);
	}

	int device, driver_Version = 0, runtime_Version = 0;

	for (device = 0; device < device_Count; ++device)
	{
		cudaSetDevice(device);
		cudaDeviceProp device_Property;
		cudaGetDeviceProperties(&device_Property, device);

		printf("\nDevice %d: \"%s\"\n", device, device_Property.name);

		// Console log
		cudaDriverGetVersion(&driver_Version);
		cudaRuntimeGetVersion(&runtime_Version);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driver_Version / 1000, (driver_Version % 100) / 10, runtime_Version / 1000, (runtime_Version % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", device_Property.major, device_Property.minor);
		printf( "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
			(float)device_Property.totalGlobalMem / 1048576.0f, (unsigned long long) device_Property.totalGlobalMem);
		printf("  (%2d) Multiprocessors", device_Property.multiProcessorCount );
		printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", device_Property.clockRate * 1e-3f, device_Property.clockRate * 1e-6f);

		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", device_Property.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n", device_Property.memoryBusWidth);
		if (device_Property.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", device_Property.l2CacheSize);
		}
		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			device_Property.maxTexture1D, device_Property.maxTexture2D[0], device_Property.maxTexture2D[1],
			device_Property.maxTexture3D[0], device_Property.maxTexture3D[1], device_Property.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			device_Property.maxTexture1DLayered[0], device_Property.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
			device_Property.maxTexture2DLayered[0], device_Property.maxTexture2DLayered[1], device_Property.maxTexture2DLayered[2]);
		printf("  Total amount of constant memory:               %lu bytes\n", device_Property.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", device_Property.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", device_Property.regsPerBlock);
		printf("  Warp size:                                     %d\n", device_Property.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", device_Property.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", device_Property.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			device_Property.maxThreadsDim[0],
			device_Property.maxThreadsDim[1],
			device_Property.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			device_Property.maxGridSize[0],
			device_Property.maxGridSize[1],
			device_Property.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", device_Property.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", device_Property.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (device_Property.deviceOverlap ? "Yes" : "No"), device_Property.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", device_Property.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", device_Property.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", device_Property.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", device_Property.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", device_Property.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", device_Property.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", device_Property.unifiedAddressing ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n", device_Property.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n", device_Property.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", device_Property.pciDomainID, device_Property.pciBusID, device_Property.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[device_Property.computeMode]);
	}	
}
