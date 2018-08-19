# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:05:29 2018

@author: bhaumik
"""


import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy
import time

n=10
a=numpy.float32(numpy.random.randint(1,5,(n,n)))
b=numpy.float32(numpy.random.randint(1,5,(n,n)))

       
tic=time.time()
axb=a*b

#print(numpy.dot(a,b))
toc=time.time()-tic
print("Dot Product on CPU")
print(toc,"s")


start = drv.Event()
end=drv.Event()
start.record()
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
axbGPU = gpuarray.dot(a_gpu,b_gpu)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Dot Product on GPU")
print("%fs" % (secs))
if(numpy.sum(axb)==axbGPU.get()):
    print("The computed dor product is correct")
    

