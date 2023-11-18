import numpy as np
import copy
a = np.array([[0.3,0.5,0.2],
                [0.2,0.6,0.2],
                [0.1,0.4,0.6]])
b = np.array([[0.1,0.5,0.2,0.1],
                [0.2,0.1,0.2,0.1],
                [0.3,0.1,0.3,0.1],
                [0.4,0.3,0.3,0.7]])

# a = a.T
# b = b.T

temp = np.array([[2,3,5]])
for i in range(50):
    temp = np.dot(temp,a)
    if i%10 == 0:
        print(temp)

