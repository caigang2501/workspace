
import math
import numpy as np
from keras import datasets
import matplotlib.pyplot as plt


(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = x.reshape(60000,784)
x = 2*x/255 - 1
x_val = x_val.reshape(10000,784)
print(x.shape,x_val.shape)

nodess = [len(x[0]),8,10]
layers = []
bss = []
i = 0
while i < len(nodess)-1:
    wss = np.random.randn(nodess[i],nodess[i+1])*np.sqrt(1 / nodess[i+1])
    layers.append(wss)
    bss.append(np.random.rand(nodess[i+1])*0.1)
    i += 1

w00 = layers[0][400].copy()
w00temp = layers[0][400].copy()

lr = 0.1

def flow_forward(data:np.array,layers):
    oss = [data]
    i = 0
    for layer in layers:
        z = np.dot(data,layer) + bss[i]
        os = 1/(np.exp(-z)+1)
        data = os
        oss.append(os)
        i += 1

    return oss

def flow_back(lr:float,layers:list,datas,n:int):
    wgtemp = np.zeros(nodess[-1])
    wgtemp[n] = 1
    wgtemp -= datas[-1]
    i = 1
    
    while i <= len(bss):
        gbs = datas[-i] * (1-datas[-i]) * wgtemp
        wgtemp = np.dot(layers[-i],gbs)
        bss[-i] = bss[-i] + lr*gbs
        layers[-i] += np.array([datas[-i-1]]).T * np.array([gbs]) * lr    
        i += 1

e = 0
while e<100:
    i = 0
    accuracyrate = 0
    while i < len(x):
        # plt.imshow(x[i].reshape(28,28))
        # plt.show()
        datas = flow_forward(x[i],layers)
        
        flow_back(lr,layers,datas,y[i])
        if i > 59000:
            if list(datas[-1]).index(max(datas[-1])) == y[i]:
                accuracyrate += 1

        i += 1
    
    print(f'e:{e}',': ',accuracyrate)
    accuracyrate = 0
    
    print(layers[0][400]-w00temp)
    w00temp = layers[0][400].copy()
    e += 1



