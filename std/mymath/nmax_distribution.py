import matplotlib.pyplot as plt
import random

def n_max_dtb(n,x=30,repetition=200000):
    # y = [0]*x
    # for i in range(repetition):
    #     y[max(list(random.randint(0,x-1) for i in range(n)))] += 1
    # return y

    return [n*e**(n-1)/x**n for e in range(x)]



def drow(x,*args):
    fig,ax = plt.subplots()
    for y in args:
        ax.plot(range(x),y)
    plt.show()

drow(30,n_max_dtb(2),n_max_dtb(3),n_max_dtb(4),n_max_dtb(5),n_max_dtb(6),n_max_dtb(7))