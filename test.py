dic1 = {'a':1}
dic2 = {'a':1,'b':1}
dic2['b'] -= 1
p = dic2.pop('b')
print(dic1==dic2,p)