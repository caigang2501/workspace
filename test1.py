from collections import deque
import sys

queue = deque([1,2,3,4])
queue.rotate()
print(len(queue))