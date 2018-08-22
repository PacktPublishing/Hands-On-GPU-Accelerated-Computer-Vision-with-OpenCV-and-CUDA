# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:15:39 2018

@author: bhaumik
"""

import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import pycuda.autoinit

mod = SourceModule \
    (
        """
#include<stdio.h>
#define INDEX(a, b) a*256+b

__global__ void bgr2gray(float *d_result,float *b_img, float *g_img, float *r_img)
{
unsigned int idx = threadIdx.x+(blockIdx.x*(blockDim.x*blockDim.y));

unsigned int a = idx/256;
unsigned int b = idx%256;
d_result[INDEX(a, b)] = (0.299*r_img[INDEX(a, b)]+0.587*g_img[INDEX(a, b)]+0.114*b_img[INDEX(a, b)]);

}
  
  """
      )

cap = cv2.VideoCapture(0)
bgr2gray = mod.get_function("bgr2gray")
while(True):
    # Capture frame-by-frame
    ret, h_img = cap.read()
    h_img = cv2.resize(h_img,(256,256),interpolation = cv2.INTER_CUBIC)

    b_img = h_img[:, :, 0].reshape(65536).astype(np.float32)
    g_img = h_img[:, :, 1].reshape(65536).astype(np.float32)
    r_img = h_img[:, :, 2].reshape(65536).astype(np.float32)
    h_result=r_img
   
    bgr2gray(drv.Out(h_result), drv.In(b_img), drv.In(g_img),drv.In(r_img),block=(1024, 1, 1), grid=(64, 1, 1))

    h_result=np.reshape(h_result,(256,256)).astype(np.uint8)
    cv2.imshow("Grayscale Image",h_result)

    # Display the resulting frame
    cv2.imshow('Original frame',h_img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

