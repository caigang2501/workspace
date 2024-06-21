
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
    def totalMoney(self, n: int) -> int:
        weeks = n//7
        left_day = n%7

        ans = 0
        ans += 7*weeks*(weeks+1)//2+weeks*21
        ans += (weeks+1+weeks+1+left_day-1)*left_day//2
        return ans
s = Solution()
l = [-4,-2,-3]
print(s.totalMoney(4))


