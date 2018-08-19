# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 13:13:51 2018

@author: bhaum
"""

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy
from pycuda.scan import InclusiveScanKernel
import pycuda.autoinit
n=10
start = drv.Event()
end=drv.Event()
start.record()
kernel = InclusiveScanKernel(numpy.uint32,"a+b")
h_a = numpy.random.randint(1,10,n).astype(numpy.int32)
d_a = gpuarray.to_gpu(h_a)
kernel(d_a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
assert(d_a.get() == numpy.cumsum(h_a,axis=0)).all()
print("The input data:")
print(h_a)
print("The computed cumulative sum using Scan:")
print(d_a.get())
print("Cumulative Sum on GPU")
print("%fs" % (secs))

   