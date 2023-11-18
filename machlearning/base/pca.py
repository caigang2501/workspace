import numpy as np

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


