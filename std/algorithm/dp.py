import numpy as np


def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)

    # 创建一个二维数组用于存储子问题的解
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 填充 dp 数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # 构造最长公共子序列
    print(np.array(dp))
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 反转得到正常顺序
    lcs = lcs[::-1]
    
    return lcs

def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]

if __name__=='__main__':
    # X = "bdcaba"
    # Y = "abcbdab"
    # result = longest_common_subsequence(X, Y)
    # print(result)

    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    print("最大价值:", knapsack(weights, values, capacity))  # 输出最大价值
