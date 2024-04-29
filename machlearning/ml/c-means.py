import numpy as np

def cmeans(data, c, m, max_iter=100, tol=1e-4):
    # 初始化隶属度矩阵
    n = data.shape[0]
    u = np.random.rand(n, c)
    div = np.sum(u, axis=1, keepdims=True)
    u = u / div

    # 迭代更新聚类中心和隶属度
    for _ in range(max_iter):
        # 更新聚类中心
        a = np.dot(u.T**m, data)
        b = np.sum(u**m, axis=0, keepdims=True).T
        centroids = a/b
        
        # 计算距离并更新隶属度
        dist = np.linalg.norm(data[:, None, :] - centroids, axis=2)
        u_new = 1 / np.sum((dist[:,:,None] / dist[:,None,:])**(2/(m-1)), axis=2)
        
        # 计算收敛性
        if np.linalg.norm(u_new - u) < tol:
            break
        
        u = u_new
    
    return centroids, u

# 生成随机数据
np.random.seed(42)
data = np.random.randn(100, 2)

# 使用自己实现的C-means算法聚类
centroids, u = cmeans(data, c=3, m=2)

# 打印聚类中心
print("聚类中心：")
print(centroids)

# 打印隶属度矩阵
print("\n隶属度矩阵：")
print(u)
