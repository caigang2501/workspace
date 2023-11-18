import pandas as pd

# 创建一个示例DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1]
}
df = pd.DataFrame(data)

# 对每列应用内置的聚合函数
# result = df.agg(['sum', 'mean', 'min', 'max'])
# 对特定列应用多个聚合函数
result = df['A'].agg(['sum', 'mean', lambda x: x.max() - x.min()])
print(result)
