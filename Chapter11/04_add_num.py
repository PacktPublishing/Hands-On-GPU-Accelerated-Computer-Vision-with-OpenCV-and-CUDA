# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:04:33 2018

@author: bhaumik
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
mod = SourceModule("""
                   #include <stdio.h>
__global__ void add_num(float *d_result, float *d_a, float *d_b)
{
 const int i = threadIdx.x;  
 d_result[i] = d_a[i] + d_b[i];
}
""")

add_num = mod.get_function("add_num")

h_a = numpy.random.randn(1).astype(numpy.float32)
h_b = numpy.random.randn(1).astype(numpy.float32)

h_result = numpy.zeros_like(h_a)
d_a = drv.mem_alloc(h_a.nbytes)
d_b = drv.mem_alloc(h_b.nbytes)
d_result = drv.mem_alloc(h_result.nbytes)
drv.memcpy_htod(d_a,h_a)
drv.memcpy_htod(d_b,h_b)

add_num(
        d_result, d_a, d_b,
        block=(1,1,1), grid=(1,1))
drv.memcpy_dtoh(h_result,d_result)
print("Addition on GPU:")
print(h_a[0],"+", h_b[0] , "=" , h_result[0])