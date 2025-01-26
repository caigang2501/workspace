from std.algorithm.tree_ import TreeNode,list2tree,bfs_by_layer
from typing import List,Optional
from collections import deque
from heapq import heappush,heappop,heappushpop,heapify
from collections import defaultdict

class Solution:
    def connect(self, root: Optional[TreeNode]) -> int:

        pass

s = Solution()
root = list2tree([1,7,0,7,-8,None,None])
result = bfs_by_layer(root)
print(result)