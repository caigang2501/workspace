import matplotlib.pyplot as plt
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import inspect

def plot_distribution(getrandom,*args,num_samples=10000):
    samples = [getrandom(*args) for _ in range(num_samples)]
    # samples = [i for i in samples if i<70]
    print(samples[0])
    print(min(samples),max(samples))
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='b')
    plt.title('Random Number Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def getrandom(n):
    return sum([(2*random.random()-1)*(2*random.random()-1) for _ in range(n)])

def __get_delta():
    u = random.random()
    if u < 0.5:
        return (2 * u) ** (1 / (5 + 1)) - 1 # self.mutation_param = 5
    return 1 - (2 * (1 - u)) ** (1 / (5 + 1))


def plot_function_graph(function, *args):
    """
    自动绘制二维或三维函数图像。

    参数:
        function: 要绘制的函数 (单变量或双变量)
        *args: 参数取值范围
               - 对于单变量函数：args = ((a, b),)
               - 对于双变量函数：args = ((a, b), (c, d))
    """
    # 检查函数参数数量
    param_count = len(inspect.signature(function).parameters)

    if param_count == 1 and len(args) == 1:
        # 绘制二维曲线
        a, b = args[0]
        x = np.linspace(a, b, 500)
        y = function(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=f'{function.__name__}(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('2D Function Plot')
        plt.grid(True)
        plt.legend()
        plt.show()

    elif param_count == 2 and len(args) == 2:
        # 绘制三维曲面
        a, b = args[0]
        c, d = args[1]
        x = np.linspace(a, b, 100)
        y = np.linspace(c, d, 100)
        X, Y = np.meshgrid(x, y)
        Z = function(X, Y)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('3D Function Plot')
        plt.show()

    else:
        raise ValueError("函数参数数量与 args 的范围不匹配！")
import math 
def f1(x):
    return np.log(x)*np.log(1-x)

# def f2(x, y):
#     return np.sin(np.sqrt(x**2 + y**2))

if __name__=='__main__':
    # plot_distribution(np.random.rand)
    plot_function_graph(f1, (0, 1))

    # plot_function_graph(f2, (-5, 5), (-5, 5))



