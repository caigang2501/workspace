
from collections import deque
from queue import Queue


#deQue
llist = deque('12345')
llist.rotate()
print(llist)


#queue
queue_obj = Queue()  
for i in range(4):
    queue_obj.put(i)
while not queue_obj.empty():
    print(queue_obj.get())


