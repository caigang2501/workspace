# subdirectory/sub_module.py

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


a = set([1,2])
b = set([1,2,3,4])

print(b>a)