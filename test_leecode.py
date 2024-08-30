
from typing import List,Optional
from collections import deque
from heapq import heappush,heappop,heappushpop,heapify
from collections import defaultdict

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stc = []
        ans = [0]*len(nums)
        for i,n in enumerate(nums):
            while stc and stc[-1]<n:
                curr = stc.pop()
                ans[curr[0]] = n
            stc.append((i,n))
        for i,n in enumerate(nums):
            while len(stc)>1 and stc[-1]<n:
                curr = stc.pop()
                ans[curr[0]] = n
            if len(stc)==1:
                ans[stc[0][0]] = -1
                break
        return ans


# s = Solution()
# l = [-4,-2,-3]
# print(s.totalMoney(4))

class DdNode():
    def __init__(self,key=0,val=0):
        self.abov = None
        self.next = None
        self.key = key
        self.val = val
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = {}
        self.head,self.tail = DdNode(),DdNode()
        self.head.next = self.tail
        self.tail.abov = self.head

    def get(self, key: int) -> int:
        if key in self.dic:
            val = self.move2head(key)
            return val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic[key].val = value
            self.move2head(key)
        else:
            curr = DdNode(key,value)
            self.dic[key] = curr

            next = self.head.next
            self.head.next = curr
            curr.abov = self.head
            curr.next = next
            next.abov = curr
            if len(self.dic)>self.capacity:
                self.dic.pop(self.tail.abov.val)
                abov = self.tail.abov.abov
                abov.next = self.tail
                self.tail.abov = abov
            self.print_node(self.head)

        
    def move2head(self,key):
        curr = self.dic[key]
        abov = curr.abov
        next = curr.next
        abov.next = next
        next.abov = abov

        next = self.head.next
        self.head.next = curr
        curr.abov = self.head
        curr.next = next
        next.abov = curr

        return curr.val
    
    def print_node(self,head):
        curr = head.next
        t = []
        while curr:
            t.append(curr.val)
            curr = curr.next
        print(t)
        print(self.dic.keys())
        

if __name__=='__main__':
    lru = LRUCache(2)
    lru.put(2,1)
    lru.put(3,2)
    print(lru.get(2))
    # lru.put(3,3)
    # print(lru.get(2))
    # lru.put(4,4)
    # print(lru.get(1))
    # print(lru.get(3))
    # print(lru.get(4))
