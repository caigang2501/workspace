from collections import deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# 并查集
def disjoint_set(edges):
    parts = []
    for edge in edges:
        temp = []
        for i in range(len(parts)):
            for x in edge:
                if x in parts[i]:
                    temp.append([i,x])
        if len(temp)==1:
            edge.remove(temp[0][1])
            parts[temp[0][0]] += edge
        elif len(temp)==2:
            if temp[0][0] != temp[1][0]:
                parts[temp[0][0]] += parts[temp[1][0]]
                parts.pop(temp[1][0])
        else:
            parts.append(edge)
    return parts



def bfs_linked(i,j,graph):
    visited = [0]*len(graph)
    shortest_path = 0
    bucket = deque([i])
    visited[i] = 1
    while bucket:
        curr = bucket.popleft()
        for n in graph[curr]:
            if not visited[n]:
                visited[n] = 1
                bucket.append(n)
                if n==j:
                    return True
        shortest_path += 1
    
    return False
        
# mst
def mst_(data,plot=False):
    data = np.array(data)
    # 计算所有点之间的欧几里得距离
    dist_matrix = squareform(pdist(data, 'euclidean'))

    # 转换为稀疏矩阵
    sparse_graph = csr_matrix(dist_matrix)

    mst = minimum_spanning_tree(sparse_graph)

    # 提取最小生成树的非零元素（即边）
    mst_edges = np.array(mst.nonzero()).T

    if plot:
        plt.scatter(data[:, 0], data[:, 1], color='red')
        for edge in mst_edges:
            i, j = edge
            plt.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], 'b-')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Minimum Spanning Tree')
        plt.show()
    return mst_edges


if __name__=='__main__':
    # print(disjoint_set([[6,7],[1,2],[4,5],[2,3],[5,6]]))
    data = [(0, 0), (2, 2), (3, 10), (5, 2), (7, 0)]
    result = mst_(data,plot=True)
    print(result)