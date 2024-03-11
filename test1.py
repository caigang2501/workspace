from multiprocessing import Process
import os
from typing import List

import time

from multiprocessing import Process, Pipe

def f(conn=2):
    print(conn)

if __name__ == '__main__':
    p = Process(target=f)
    p.start()
    p.join()
