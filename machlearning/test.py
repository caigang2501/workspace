
from array import array
import numpy as np
import math
import queue
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

# que = queue.Queue()

# for i in range(5):
#     que.put(i)

# while not que.empty():
#     print(que.get())

# with open("testt.txt", "wt",encoding='UTF-8') as out_file:
#     out_file.write(f"该文本会写入到文件中\n看到我了吧！{a.count(1)}")
# with open("testt.txt", "a",encoding='UTF-8') as out_file:
#     out_file.write(f"该文本会写入到文件中\n看到我了吧！{len(a)}")


# y_onehot = np.zeros((1, 10))
# y_onehot[np.arange(1), 4] = 1
# print(y_onehot)
# a = np.array([np.linspace(1,4,4)])
# b = np.array([np.linspace(1,5,5)])
# print(a.shape,b.shape)
# print(np.dot(a.T,b))
# print(a*b.T)
# print(a.T*b)

# a = np.ones((3,4))
# print(a)

a = np.array([[-1,0,0],[0,-1,0],[0,0,5]])
x = np.array([[1,0,1],[0,1,1],[-1,-1,1]])
# x = np.array([[0,1,1],[1,0,1],[-1,-1,1]])
print(np.dot(np.dot(np.linalg.inv(x),a),x))