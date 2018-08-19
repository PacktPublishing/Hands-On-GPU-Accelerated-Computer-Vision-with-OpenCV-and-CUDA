# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 11:29:01 2018

@author: bhaumik
"""

import pycuda.driver as drv
import pycuda.autoinit
drv.init()
print("%d device(s) found." % drv.Device.count())
for i in range(drv.Device.count()):
    dev = drv.Device(i)
    print("Device #%d: %s" % (i, dev.name()))
    print("  Compute Capability: %d.%d" % dev.compute_capability())
    print("  Total Memory: %s GB" % (dev.total_memory()//(1024*1024*1024)))
    attributes = [(str(prop), value) 
            for prop, value in list(dev.get_attributes().items())]
    attributes.sort()
    n=0
    for prop, value in attributes:
        print("  %s: %s " % (prop, value),end=" ")
        n = n+1
        if(n%2 == 0):
            print(" ")




