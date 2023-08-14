# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:45:31 2022

@author: dell
"""
import tensorflowrtest as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import make_moons
from sklearn.model_selection import train_test_split

# def himmelblau(x):
#  # himmelblau 函数实现，传入参数 x 为 2 个元素的 List
#      return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# x = tf.constant([4., 0.]) # 初始化参数
# for step in range(200):# 循环优化 200 次
#     with tf.GradientTape() as tape: #梯度跟踪
#         tape.watch([x]) # 加入梯度跟踪列表
#         y = himmelblau(x) # 前向传播
#     # 反向传播
#     grads = tape.gradient(y, [x])[0]
#     x -= 0.01*grads
#     # 打印优化的极小值
#     if step % 20 == 19:
#         print ('step {}: x = {}, f(x) = {}'
#         .format(step, x.numpy(), y.numpy()))


N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数量比率
# 利用工具函数直接生成数据集
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100) 
# 将 2000 个点按着 7:3 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)





