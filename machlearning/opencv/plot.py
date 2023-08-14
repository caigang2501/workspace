# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:38:39 2022

@author: dell
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = np.zeros((255,255,3),np.uint8)

cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.circle(img,(256,256),60,(0,0,255),-1)
cv.rectangle(img,(100,100),(400,400),(0,255,0),5)

plt.imshow(img[:,:,::-1])
plt.show()