# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:05:29 2018

@author: bhaumik
"""
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy
from pycuda.reduction import ReductionKernel
import pycuda.autoinit
n=5
start = drv.Event()
end=drv.Event()
start.record()
d_a = gpuarray.arange(n,dtype= numpy.uint32)
d_b = gpuarray.arange(n,dtype= numpy.uint32)
kernel = ReductionKernel(numpy.uint32,neutral="0",reduce_expr="a+b",map_expr="d_a[i]*d_b[i]",arguments="int *d_a,int *d_b")
d_result = kernel(d_a,d_b).get()
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Vector A")
print(d_a)
print("Vector B")
print(d_b)
print("The computed dot product using reduction:")
print(d_result)
print("Dot Product on GPU")
print("%fs" % (secs))

    
    

