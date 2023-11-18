import pandas as pd
try:
    number = int(input("请输入数字："))
    print("number:",number)
    print("=======hello======")
except Exception as e:
    # 报错错误日志
    print("打印异常详情信息： ",e)
else:
    print("没有异常")
finally:#关闭资源
    print("finally")
print("end")

