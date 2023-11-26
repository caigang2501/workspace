import numpy as np
# 将坐标轴中心移到数据的中心，然后旋转坐标轴，使得数据在C1轴上的方差最大，
# 即全部n个数据个体在该方向上的投影最为分散。意味着更多的信息被保留下来。C1成为第一主成分。
# C2第二主成分：找一个C2，使得C2与C1的协方差（相关系数）为0，以免与C1信息重叠，并且使数据在该方向的方差尽量最大。
# data = np.array([[1, 2, 3,6],
#                  [4, 5, 6,7],
#                  [7, 8, 9,3],
#                  [10,12,13,2],
#                  [10,13,13,9],
#                  [10,14,13,8],])
data = np.array([[1,1,1],
                 [2,2,2],
                 [3,3,3]])
mean = np.mean(data, axis=0)
centered_data = data - mean
print('\nmean:\n',mean,'\ncentered_data:\n',centered_data)

cov_matrix = np.cov(centered_data, rowvar=False)
print('cov_matrix:\n',cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print('\neigenvalues:\n',eigenvalues,'\neigenvectors:\n',eigenvectors)
k = 2  # 选择保留2个主成分作为例子
top_k_eigenvectors = eigenvectors[:, :k]
projected_data = np.dot(centered_data, top_k_eigenvectors)

print('\ntop_k_eigenvectors:\n',top_k_eigenvectors,'\nprojected_data:\n',projected_data)


