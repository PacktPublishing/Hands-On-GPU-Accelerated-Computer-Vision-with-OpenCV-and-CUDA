# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:48:28 2018

@author: bhaumik
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
N = 1
from pycuda.compiler import SourceModule
mod = SourceModule("""
                   
__global__ void add_num(float *d_result, float *d_a, float *d_b)
{
 const int i = threadIdx.x;  
 d_result[i] = d_a[i] + d_b[i];
}
""")
add_num = mod.get_function("add_num")
h_a = numpy.random.randn(N).astype(numpy.float32)
h_b = numpy.random.randn(N).astype(numpy.float32)
h_result = numpy.zeros_like(h_a)
add_num(
        drv.Out(h_result), drv.In(h_a), drv.In(h_b),
        block=(N,1,1), grid=(1,1))
print("Addition on GPU:")
for i in range(0,N):
    print(h_a[i],"+", h_b[i] , "=" , h_result[i])