import pandas as pd
import  numpy as np


# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'],index=["a","b","c"])
# print(df)

# data = {'name':['Tom', 'Jack', 'Steve', 'Ricky'],'age':[28,34,29,42]}
# df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
# df.insert(1,column='score',value=[91,90,75,89])
# print(df)
# df = df.drop('rank1')
# del(df['name'])
# print(df[0:2])
# df.pop('name')
# print(df.score[1])
# print(df.T)

# df = pd.DataFrame(np.arange(9).reshape(3,3),columns=['c1','c2','c3'])
# print(df)
# print(df.apply(lambda x: x.max() - x.min()))
# print(df.apply(lambda x: x.max() - x.min(),axis=1))

# 将第一列的所有值设置为 1
# df.iloc[:, 0] = 1

# df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
# df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
# result = pd.merge(df1,df2,on='key',how='left') # how = inner,left,right,outer

# result_vertical = pd.concat([df1, df2])


df1 = pd.DataFrame(np.arange(12).reshape(4,3),columns=['c1','c2','c3'])
df2 = pd.DataFrame(np.arange(9).reshape(3,3),columns=['c1','c2','c3'])
# df1.loc[df1['c1'].isin(df2['c1']),'c1'] = 100
df1.loc[df1['c1']%2==1,'c1'] = 100
# df2.insert(1,'cin',df1['c2'])
df2['c4'] = [1,2,3]
# print(type(df2['c1']),type(df2.loc[1]),type(df2.loc[1,'c1']),df1.columns)
# print(pd.concat([df1, df2]))
# print(type(df1['c1'].values),df1['c1'].values)

data_list = df1.values.tolist()


data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 15, 20, 25, 30],
    'Value1': [10, 15, 20, 25, 30]
}
df = pd.DataFrame(data)


df3 = df.groupby('Category').agg({'Value':'min','Value1':'count'})
# df3 = df.groupby('Category')['Value'].transform('count')

for index, row in df.iterrows():
    print(row.values)


