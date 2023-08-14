import matplotlib.pyplot as plt
import numpy as np
import random
import math

#中心极限定理，均匀分布，n = 4
def test1():

    # l = []
    # for i in range(1000000):
    #     l.append(random.randrange(21)+random.randrange(21)+random.randrange(21))

    n = 4
    rng = np.random.default_rng(12345)
    l = [0]*1000000
    for i in range(n):
        rints = rng.integers(low=0, high=21, size=1000000)
        l += rints

    x = np.linspace(0,n*20,n*20+1)
    y = [0]*(n*20+1)
    for i in l:
        y[i] += 1

    return x,y

#二项分布
def test2():
    x = np.linspace(0,40,41)
    y = []
    for i in range(41):
        y.append(math.factorial(41)/math.factorial(i)/math.factorial(41-i))

    return x,y

# np.random.default_rng为均匀分布
def test3():
    x = np.linspace(0,40,41)

    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=41, size=1000000)
    y = [0]*41
    for i in rints:
        y[i] += 1
    

    # y = [0]*10
    # for i in range(2):
    #     rints = rng.integers(low=-10, high=10, size=10)
    #     rints *= rints
    #     y += rints
   
    return x,y

#正态分布np.random.normal(expectation,variance,size)
def test4():
    n = 1000000
    l = np.random.normal(20,5,n)//1

    x = np.linspace(0,40,41)
    y = [0]*(41)
    for i in l:
        if 0<i<40:
            y[int(i)] += 1
    print(l[:15])
    print(y[-10:])

    return x,y

#1到100中取n个数的最大一个数的f(x)和F(x)
def test5(n):
    rng = np.random.default_rng(12345)
    l = []
    rints = rng.integers(low=1, high=101, size=n*500000)
    for i in range(0,n*500000,n):
        maxint = max(rints[i:i+n])
        l.append(maxint)

    x = np.linspace(1,100,101)
    y = [0]*(101)
    for i in l:
        y[i] += 1

    return x,y

def drow(t,t1 = None,t2 = None):
    fig,ax = plt.subplots()
    x,y = t
    ax.plot(x,y)

    if t1 != None:
        x,y = t1
        ax.plot(x,y)
    if t2 != None:
        x,y = t2
        ax.plot(x,y)
    plt.show()

drow(test5(2),test5(3),test5(4))

