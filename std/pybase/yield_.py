def my_generator():
    yield 1
    print('run yield 2')
    yield 2
    print('run yield 3')
    yield 3

# 创建生成器对象
gen = my_generator()

# 通过迭代获取生成器产生的值
print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
print(next(gen))  # 输出: 3