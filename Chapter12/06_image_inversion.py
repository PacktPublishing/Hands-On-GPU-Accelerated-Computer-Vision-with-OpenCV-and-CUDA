# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:34:15 2018

@author: bhaumik
"""

import pycuda.driver as drv
import numpy as np
import cv2
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
  
img = cv2.imread('circles.png',0)

#print a
h_img = img.reshape(65536).astype(np.float32)
d_img = gpuarray.to_gpu(h_img)
d_result = 255- d_img
h_result = d_result.get()
h_result=np.reshape(h_result,(256,256)).astype(np.uint8)
cv2.imshow("Image after Inversion",h_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
