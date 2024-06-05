import glob
from mimetypes import init
import sys
import numpy as np
import os

# 字典
d = {'a':1,'b':2,'c':3}
d.update({'d':4})
d.setdefault('e',5)
d.pop('a')
# print(d.get('a',-1))
# print(d)

a = ['a','b','c']
b = [1,2,3]
dic = dict(zip(a,b))
# print(dic)

# list
a = [b for b in range(10) if b%2==0]
print(a)