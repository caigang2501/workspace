
from typing import List,Optional
from collections import deque
from heapq import heappush,heappop,heappushpop,heapify

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        r = max(nums)-min(nums)
        t = r*k//len(nums)
        lst = []
        while len(lst)<k:
            lst = []
            for i in nums:
                if t<i:
                    lst.append(i)
            t -= 1 if r<10 else r//10
        return self.heap_k(lst,k)
        
    def heap_k(self,nums,k):
        hp = nums[:k]
        heapify(hp)
        for i in nums[k:]:
            heappushpop(hp,i)
        return heappop(hp)    

s = Solution()
l = [3,2,1,5,6,4]
result = s.findKthLargest(l,2)
print(result)
