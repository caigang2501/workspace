
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
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        sum_turn = (1+num_people)*num_people//2

        turn = candies//sum_turn
        left = candies%sum_turn
        ans = [(i+1)*turn if (1+i)*i//2<=sum_turn else i*turn for i in range(1,num_people+1)]
        return ans
s = Solution()
l = [-4,-2,-3]
print(s.distributeCandies(10,3))


