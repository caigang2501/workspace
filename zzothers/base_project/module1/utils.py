import time
from module1 import constent as c
from flask import g,session             # g:独立全局变量  session:共享全局变量
import redis                            # 缓存


def change():
    for i in range(5):
        time.sleep(1)
        c.param += 1
        c.lst.append(i)
        g.some_value = 2
        print(i)

