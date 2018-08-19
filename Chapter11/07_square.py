# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 08:55:00 2018

@author: bhaumik
"""

import pycuda.driver as drv
import pycuda.autoinit  
from pycuda.compiler import SourceModule
import numpy
mod = SourceModule("""
    __global__ void square(float *d_a)
    {
      int idx = threadIdx.x + threadIdx.y*5;
      d_a[idx] = d_a[idx]*d_a[idx];
    }
    """)
start = drv.Event()
end=drv.Event()
h_a = numpy.random.randint(1,5,(5, 5))
h_a = h_a.astype(numpy.float32)
h_b=h_a.copy()
start.record()
d_a = drv.mem_alloc(h_a.size * h_a.dtype.itemsize)
drv.memcpy_htod(d_a, h_a)
square = mod.get_function("square")
square(d_a, block=(5, 5, 1), grid=(1, 1), shared=0)
h_result = numpy.empty_like(h_a)
drv.memcpy_dtoh(h_result, d_a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU without inout")
print("%fs" % (secs))
print("original array:")
print(h_a)
print("Square with kernel:")
print(h_result)

#----Using inout functionality of driver class -------------------------------------------------
start.record()
start.synchronize()
square(drv.InOut(h_a), block=(5, 5, 1))
end.record()
end.synchronize()

print("Square with InOut:")
print(h_a)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with inout")
print("%fs" % (secs))

# ---------------Using gpuarray class----------------------------------------------------#

import pycuda.gpuarray as gpuarray
start.record()
start.synchronize()
h_b = numpy.random.randint(1,5,(5, 5))
#h_b = h_b.astype(numpy.float32)
d_b = gpuarray.to_gpu(h_b.astype(numpy.float32))
h_result = (d_b**2).get()
end.record()
end.synchronize()
print("original array:")
print(h_b)
print("Squared with gpuarray:")
print(h_result)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with gpuarray")
print("%fs" % (secs))