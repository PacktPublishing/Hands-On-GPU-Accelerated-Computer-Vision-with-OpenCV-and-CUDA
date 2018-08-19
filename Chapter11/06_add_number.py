# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:39:19 2018

@author: bhaumik
"""
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
import math
N = 1000000

from pycuda.compiler import SourceModule
mod = SourceModule("""
                   
__global__ void add_num(float *d_result, float *d_a, float *d_b,int N)
{
 int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	while (tid < N)
    {
 d_result[tid] = d_a[tid] + d_b[tid];
 tid = tid + blockDim.x * gridDim.x;
}
    }
""")
start = drv.Event()
end=drv.Event()
add_num = mod.get_function("add_num")

h_a = numpy.random.randn(N).astype(numpy.float32)
h_b = numpy.random.randn(N).astype(numpy.float32)

h_result = numpy.zeros_like(h_a)
h_result1 = numpy.zeros_like(h_a)
n_blocks = math.ceil((N/1024) +1)
start.record()
add_num(
        drv.Out(h_result), drv.In(h_a), drv.In(h_b),numpy.uint32(N),
        block=(1024,1,1), grid=(n_blocks,1))
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%N)
print("%fs" % (secs))
start = time.time()
for i in range(0,N):
    h_result1[i] = h_a[i] +h_b[i]
end = time.time()
print("Addition of %d element of CPU"%N)
print(end-start,"s")
