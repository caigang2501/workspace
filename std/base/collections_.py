
from collections import deque,Counter
from queue import Queue,PriorityQueue
from heapq import heappop,heappush,heapify
from typing import List
#deQue 队列两端高效地进行插入和删除操作
def test_deque():
    llist = deque('12345')
    llist.append('6')
    llist.appendleft('0')
    print(llist)


#queue 先进先出
def test_queue():
    que = Queue()  
    for i in range(4):
        que.put(i)
    while not que.empty():
        print(que.get())

# PriorityQueue
def test_PriorityQueue():
    my_priority_queue = PriorityQueue()
    my_priority_queue.put((1, "Priority 1"))
    my_priority_queue.put((3, "Priority 3"))
    my_priority_queue.put((2, "Priority 2"))
    element = my_priority_queue.get()
    print(element)  # 输出: (1, 'Priority 1')

#Counter
def test_counter():
    l = [1,1,2,3,3,4]
    print(Counter(l))

# 单调栈 每个元素的下一个更大元素位置
def monotonic_stack(nums):
    stack = []  # 单调递增栈，存储元素的索引
    result = [-1] * len(nums)  # 存储结果，初始化为-1

    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)

    return result

def heaptest():
    a = [8, 2, 6, 9, 3, 1, 4, 7]
    heapify(a)  
    heappush(a,5)
    ans = []
    while a:
        ans.append(heappop(a))
    return ans

if __name__=='__main__':
    r = heaptest()
    # test_counter()
    # print(monotonic_stack([2, 1, 2, 4, 3]))
    print(r)