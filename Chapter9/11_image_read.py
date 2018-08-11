# -*- coding: utf-8 -*-

import numpy as np
import cv2
img = cv2.imread('images/cameraman.tif',0)

cv2.imshow("Image read in Python", img)
k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
     cv2.destroyAllWindows()

