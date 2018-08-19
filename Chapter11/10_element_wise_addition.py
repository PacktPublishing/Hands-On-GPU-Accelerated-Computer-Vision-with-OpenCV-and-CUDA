# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:13:27 2018

@author: bhaumik
"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel

add = ElementwiseKernel(
        "float *d_a, float *d_b, float *d_c",
        "d_c[i] = d_a[i] + d_b[i]",
        "add")

# create a couple of random matrices with a given shape
from pycuda.curandom import rand as curand
shape = 1000000
d_a = curand(shape)
d_b = curand(shape)
d_c = gpuarray.empty_like(d_a)
start = drv.Event()
end=drv.Event()
start.record()
add(d_a, d_b, d_c)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%shape)
print("%fs" % (secs))
# check the result
if d_c == (d_a + d_b):
    print("The sum computed on GPU is correct")

