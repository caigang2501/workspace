# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:23:52 2022

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt

data = 2*np.random.rand(1000,2)-1
print(data[0])
x = data[:,0]
y = data[:,1]
idx = x**2 + y**2 < 1
print(idx[0:10])
hole = x**2 + y**2 > 0.25
idx = ~np.logical_xor(idx,hole)
plt.plot(x[idx], y[idx],'ro',markersize=1)
plt.show()