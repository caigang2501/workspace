import os

#@property将方法转为属性，以属性方式调用方法

#===================scope=======================

class A:
    def __init__(self) -> str:
        global out #全局变量
        out = 2
        print(out)  # 可以读外部变量
        # funtest1() 不可访问外部函数


    c = 1
    __b = 3
    def funtest2(self):
        print(self.c)

    def funtest3(self):
        print(self.a)



class B(A):
    b = 2

class C:
    def __init__(self,A) -> str:
        print('c')
    b = 2

# a = A()
# b = B()
# print(out)

a = 3
def funtest1():
    # global a
    # a += 1
    print(a) 

# funtest1()
# print(a)


#===================__new__=======================
class NA:
    
    b = 3
    def __new__(cls):
        a = 'new'
        print(a)
        return super().__new__(cls)

    def __init__(self):
        a = 'init'
        print(a)

    def fun():
        pass

# a = NA()
# print(NA())



tup = (1,2,[3,4])
try:
    tup[2]+=[5,6]
except Exception:
    pass
print(tup)

#===================__init__=======================
#子类对象创建时不会调父类的__nint__
class Animal:

    def __init__(self,color="白色"):

        Animal.color = color

    def get_color(self):
        print("Animal的颜色为",Animal.color)
class Cat(Animal):

    def __init__(self):
        pass

cat = Cat()

cat.get_color()
