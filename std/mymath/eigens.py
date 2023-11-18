import numpy as np
from scipy.linalg import eig

# 创建一个示例矩阵
A = np.array([[0.8, -0.6],
              [0.6, 0.8]])
# A = np.array([[0.3,0.5,0.2],
#                 [0.2,0.6,0.2],
#                 [0.1,0.4,0.6]])
# 计算特征值和特征向量
eigenvalues, eigenvectors = eig(A)

print("特征值：")
print(eigenvalues)

print("特征向量：")
print(eigenvectors)
