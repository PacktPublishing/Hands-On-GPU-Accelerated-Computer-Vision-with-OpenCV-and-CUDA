# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:16:59 2018

@author: bhaumik
"""

import numpy as np
from pycuda import driver, gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
MATRIX_SIZE = 3  
matrix_mul_kernel = """
__global__ void Matrix_Mul_Kernel(float *d_a, float *d_b, float *d_c)
{
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      float value = 0;
  
      for (int i = 0; i < %(MATRIX_SIZE)s; ++i) {
          float d_a_element = d_a[ty * %(MATRIX_SIZE)s + i];
          float d_b_element = d_b[i * %(MATRIX_SIZE)s + tx];
           value += d_a_element * d_b_element;
       }
 
       d_c[ty * %(MATRIX_SIZE)s + tx] = value;
   } """
  
matrix_mul = matrix_mul_kernel % {'MATRIX_SIZE': MATRIX_SIZE}
  
mod = SourceModule(matrix_mul)
  
h_a = np.random.randint(1,5,(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
h_b = np.random.randint(1,5,(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
  
  # compute on the CPU to verify GPU computation
h_c_cpu = np.dot(h_a, h_b)
  
  
d_a = gpuarray.to_gpu(h_a) 
d_b = gpuarray.to_gpu(h_b)
  
d_c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
  
 
matrixmul = mod.get_function("Matrix_Mul_Kernel")
  
 
matrixmul(d_a, d_b,d_c_gpu, 
      block = (MATRIX_SIZE, MATRIX_SIZE, 1),
      )
  
print("*" * 100)
print("Matrix A:")
print(d_a.get())

print("*" * 100)
print("Matrix B:")
print(d_b.get())

print("*" * 100)
print("Matrix Multiplication result:")
print(d_c_gpu.get())

if (h_c_cpu.all() == d_c_gpu.get().all()) :
    print("\n\nThe computed matrix multiplication is correct")

