# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:11:58 2022

@author: dell
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def knn_iris():
    
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state = 6)
    
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    # 注意！：x_test使用transform()方法
    x_test = transfer.transform(x_test)
    
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    
    y_predict = estimator.predict(x_test)
    print(y_predict == y_test)
    print(estimator.score(x_test,y_test))
    
    return None 

# 用朴素贝叶斯算法对新闻进行分类
def nb_news():
    
    news = fetch_20newsgroups(subset="all")
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target)
    
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    
    y_predict = estimator.predict(x_test)
    print(y_predict == y_test)
    print(estimator.score(x_test,y_test))
    
    return None 
    
    
def decissiontree_iris():
    
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size = 0.2,random_state = 22)    
    
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)
    
        
    y_predict = estimator.predict(x_test)
    print(y_predict == y_test)
    print(estimator.score(x_test,y_test))
    
    return None 
    
    
decissiontree_iris()
    
    
    
    