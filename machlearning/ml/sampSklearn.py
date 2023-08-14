# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:46:20 2022

@author: dell
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# import pandas as pd
# data = pd.read_csv("dating.txt")
# 获取前三列
# data = data.iloc[:, :3]

# from sklearn.preprocessing import MinMaxScaler,StandardScalar
# transfer = MinMaxScaler(featur_range = [2,3])
# StandardScalar()

# 过滤低方差
# from sklearn.feature_selection import VarianceThreshold
# transfer = VarianceThreshold(threshold = 5)



def datasets_demo():
    iris = load_iris()
    # print(iris)
    # print(iris.target)
    # print(iris["DESCR"])
    # print(iris.feature_names)
    # print(iris.data,iris.data.shape)
    
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size = 0.2,random_state = 22)
    print(y_test)
    return None

def dict_demo():
    
    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    
    #sparse = True时，transfer.fit_transform返回稀疏矩阵
    #sparse = False时，transfer.fit_transform矩阵
    transfer = DictVectorizer(sparse=False)
    
    data_new = transfer.fit_transform(data)
    print(data_new)
    #打印列名
    print(transfer.get_feature_names())
    
    return None

def text_demo():
    
    data = ["life is short,i like like python","life is too long i dislike python"]
    
    transfer = CountVectorizer(stop_words=["is","too"])
    data_new = transfer.fit_transform(data)
    
    #toarray()将稀疏矩阵转化为矩阵
    print(data_new.toarray())
    print(transfer.get_feature_names())

#主成分分析：PCA降维
def pca_demo():
    
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    
    # n_components取整数表示降到多少维
    # 取0~1的小数表示降到百分之多少维
    transfer = PCA(n_components=2)
    
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None    
    
pca_demo()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    