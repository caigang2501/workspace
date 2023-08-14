# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:20:43 2022

@author: dell
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img0 = cv.imread('C:opencv/tangwei01.png',1)
# img = cv.imread('C:opencv/tangwei01.png',1)

# 彩色图
# plt.imshow(img[:,:,::-1])
# # 灰度图
# plt.imshow(img,cmap=plt.cm.gray)
# plt.show()

# cv.imshow("image",img)
# cv.waitKey(0)

# 保存图像
# cv.imwrite('C:/Users/dell/py works/opencv/tangwei.png',img)


arr = np.zeros((2,2,3))
arr[:,0,0] = [255,100]
print(arr)

cv.imwrite('opencv/rgb.jpg',arr)
arr0 = cv.imread('C:opencv/rgb.jpg',0)
print(arr0)





