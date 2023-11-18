
import pandas as pd

# 假设你有两个DataFrame对象 df1 和 df2
# 例如：
df1 = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5','']})
df2 = pd.DataFrame({'B': [10, 11, 12]})

# 使用 pandas.concat 将 df1 和 df2 连接在一起
# result = pd.concat([df1, df2], ignore_index=True)
# print(result)
# for value in result['B'].values:
#     print(value,type(value))
# ignore_index=True 会重新生成索引，如果不需要保留原始索引的话

# df2['A'] = [5,6,7]
# print(df2)

df1['B'].fillna(100)
df1['C'] = df1[['A','B']].max(axis=1)
df3 = df1['A']
# df3.loc[1,'A']=33
# print(df1,'\n',df3,type(df3))










def matrix_area(a,b):
    global df1
    df1 = pd.DataFrame({'B': [10, 11, 12]})
    s = a*b
    return s

def matrix_area1(a,b):
    global df1
    s = a*b
    return df1

matrix_area(1,2)