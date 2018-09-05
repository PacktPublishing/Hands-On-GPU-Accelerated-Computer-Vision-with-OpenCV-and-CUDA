# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:25:20 2018

@author: bhaumik
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

     __global__ void myfirst_kernel()
       {
        printf("Hello,PyCUDA!!!");
      }
""")
 
function = mod.get_function("myfirst_kernel")
function(block=(1,1,1))
