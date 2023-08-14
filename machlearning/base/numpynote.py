import numpy as np

# 创建
# aa = np.zeros([2,3])
# ab = np.asarray(aa)
# ac = np.array(range(9))
# ac = ac.reshape(3,3)
# # ba = np.arange(6)
# ba = np.arange(0,12,2)
# bb = np.linspace(1,9,5)
# # 创建等比数列
# bc = np.logspace(0,9,10,base=2)
# f = np.random.randn(4,3)


# np.eye(3)#n*n单位矩阵
# np.eye(3, 5)
# np.diag([1, 2, 3])#对角矩阵
# np.diag([1, 2, 3], 1)

# 切片
# print(ac)
# print(ac[...,1])
# print(ac[[2,1,1,0]])
# print(ac[...,[2,1,1,0]])
# print(ac[...,1:])
# print(ac[np.ix_([2,1,0,2],[0,0,1,2])])
# print(ac[1:3, 1:3])
# print(ac[1:3,[1,2]])
# print(ac[[0,1,2,0],[2,2,2,1]])
# print (ac[ac >  5])

# 操作
# aa = np.zeros([2,3])
# aa = aa + 1
# aa = aa + [1,2,3]
# aa = np.array(range(6))
# a = aa.reshape(2,3)
# b = aa.reshape(3,2)
# print(a)
# print(b)
# print(np.dot(a,b))
# print(np.inner(a,b.T))
# print(np.vdot(a,b.T))

# a = np.array(range(9)).reshape(3,3)
# a[0,0] = 1
# print(a)
# print(np.linalg.det(a))
# print(np.linalg.solve(a,[1,3,2]))
# print(np.linalg.inv(a))
# for x in a.flat:
#     print(x)
# print(a.flatten("C"))#改变变量值不会影响原数组
# print(a.ravel("C"))#改变变量值会影响原数组


png = np.arange(12).reshape(3,4)
# print(png)
# print(png[0:2,0:2])
# print(np.linalg.inv(np.array([[1,2],[3,4]])))
# rows = np.array([0, 1], dtype=np.intp)
# columns = np.array([0, 2], dtype=np.intp)
# png = png[rows[:, np.newaxis],columns]
# print(png)

#高级切片
arrones = np.arange(7)
arrones = arrones[:, np.newaxis] + arrones[np.newaxis, :]
# print(arrones[0:7:2,[1,3,5]])
# print(arrones[:,[0,1,5]])
# print(arrones[:,7:8])
# # arrones[1] = [7]*3
# arrones[0:2,0:2] = np.ones((2,2))*-1
# rows = (np.arange(7) % 3) == 0
# print(arrones[rows])

#。。。。
a = np.repeat(np.arange(5).reshape([1,-1]),10,axis = 0)+10.0

