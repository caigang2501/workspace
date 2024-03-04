import numpy as np
from scipy.optimize import minimize

# 定义二次型目标函数
def quadratic_function(x, A, b, c):
    return 0.5 * x.T @ A @ x + b.T @ x + c

# 定义二次型的梯度
def quadratic_gradient(x, A, b):
    return A @ x + b

# 定义二次型的Hessian矩阵（这里假设A是对称矩阵）
def quadratic_hessian(x, A):
    return A

# 初始点
x0 = np.zeros((n,))

# 定义对称矩阵A，线性项b，常数项c
A = np.array([[2, 1], [1, 2]])
b = np.array([1, 2])
c = 3

# 使用共轭梯度法进行优化
result = minimize(quadratic_function, x0, args=(A, b, c), jac=quadratic_gradient, hess=quadratic_hessian, method='CG')

# 输出结果
print("最优解:", result.x)
print("最小值:", result.fun)
