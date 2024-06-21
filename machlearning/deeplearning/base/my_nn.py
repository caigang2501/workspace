
import math
import numpy as np
from keras import datasets
import matplotlib.pyplot as plt
from torch import nn

(x, y), (x_val, y_val) = datasets.mnist.load_data() # 28*28
x = x.reshape(60000,784)
x = x/255*2 - 1
x_val = x_val.reshape(10000,784)


class Model():
    def __init__(self,*args) -> None:
        pass

    def forward():
        pass

class Linear():
    def __init__(self,input,output) -> None:
        pass

