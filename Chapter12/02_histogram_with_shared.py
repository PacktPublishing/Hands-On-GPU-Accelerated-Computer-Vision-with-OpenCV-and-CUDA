
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:04:33 2018

@author: bhaumik
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
mod1 = SourceModule("""
__global__ void atomic_hist(int *d_b, int *d_a, int SIZE)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = blockDim.x * gridDim.x;
	__shared__ int cache[256];
	cache[threadIdx.x] = 0;
	__syncthreads();
	
	while (tid < SIZE)
	{
		atomicAdd(&(cache[d_a[tid]]), 1);
		tid += offset;
	}
	__syncthreads();
	atomicAdd(&(d_b[threadIdx.x]), cache[threadIdx.x]);
}
""")

atomic_hist = mod1.get_function("atomic_hist")
import cv2
h_img = cv2.imread("cameraman.tif",0)

h_a=h_img.flatten()
h_a=h_a.astype(numpy.int)
h_result = numpy.zeros(256).astype(numpy.int)
SIZE = h_img.size
NUM_BIN=256
n_threads= int(numpy.ceil((SIZE+NUM_BIN-1) / NUM_BIN))
start = drv.Event()
end=drv.Event()
start.record()
atomic_hist(
        drv.Out(h_result), drv.In(h_a), numpy.uint32(SIZE),
        block=(n_threads,1,1), grid=(NUM_BIN,1),shared= 256*4)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time for Calculating Histogram on GPU with shared memory")
print("%fs" % (secs))   
plt.stem(h_result)
plt.xlim([0,256])
plt.title("Histogram on GPU")



start = cv2.getTickCount()
hist = cv2.calcHist([h_img],[0],None,[256],[0,256])
end = cv2.getTickCount()
time = (end - start)/ cv2.getTickFrequency()
print("Time for Calculating Histogram using OpenCV")
print("%fs" % (secs)) 
#plt.stem(hist)
#plt.xlim([0,256])