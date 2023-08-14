import pandas as pd
import  numpy as np


# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'],index=["a","b","c"])
# print(df)

# data = {'name':['Tom', 'Jack', 'Steve', 'Ricky'],'age':[28,34,29,42]}
# df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
# df.insert(1,column='score',value=[91,90,75,89])
# print(df)
# # df = df.drop('rank1')
# # del(df['name'])
# # print(df[0:2])
# # df.pop('name')
# # print(df.score[1])
# # print(df.T)

df = pd.DataFrame(np.random.randn(4,3),columns=['c1','c2','c3'])
# print(df)
# print(df.apply(lambda x: x.max() - x.min()))
# print(df.apply(lambda x: x.max() - x.min(),axis=1))
