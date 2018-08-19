# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:21:00 2018

@author: bhaumik
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

     __global__ void myfirst_kernel()
       {
        printf("I am in block no: %d \\n", blockIdx.x);
      }
""")
 
function = mod.get_function("myfirst_kernel")
function(grid=(10,1),block=(1,1,1))