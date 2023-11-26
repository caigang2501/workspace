from collections import deque
from queue import Queue


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def list2tree(lst):
    if not lst:
        return None

    root = TreeNode(lst[0])
    queue = [root]
    i = 1

    while queue and i < len(lst):
        current = queue.pop(0)

        if lst[i] is not None:
            current.left = TreeNode(lst[i])
            queue.append(current.left)

        i += 1
        if i < len(lst) and lst[i] is not None:
            current.right = TreeNode(lst[i])
            queue.append(current.right)
        i += 1

    return root

def bfs(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        current = queue.popleft()
        result.append(current.val)

        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)

    return result

def bfs_with_None(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        current = queue.popleft()
        if current:
            result.append(current.val)
        else:
            result.append(None)
            continue

        if current.left:
            queue.append(current.left)
        else:
            queue.append(None)
        if current.right:
            queue.append(current.right)
        else:
            queue.append(None)

    return result

def dfs(root):
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        current = stack.pop()
        result.append(current.val)

        if current.right:
            stack.append(current.right)
        if current.left:
            stack.append(current.left)

    return result

def dfs_recursive(root):
    if not root:
        return []

    result = []
    result.append(root.val)

    if root.left:
        result.extend(dfs_recursive(root.left))
    if root.right:
        result.extend(dfs_recursive(root.right))

    return result

# 1 2 4 5 3 6 7
def preorder_traversal(root):
    if root is not None:
        print(root.val, end=" ")
        preorder_traversal(root.left)
        preorder_traversal(root.right)

# 4 2 5 1 6 3 7
def inorder_traversal(root):
    if root is not None:
        inorder_traversal(root.left)
        print(root.val, end=" ")
        inorder_traversal(root.right)

# 4 5 2 6 7 3 1
def postorder_traversal(root):
    if root is not None:
        postorder_traversal(root.left)
        postorder_traversal(root.right)
        print(root.val, end=" ")

#=======================tools=================================

def tree2list(root):
    stack = [root]
    l = []
    while stack:
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
        if not node.left and not node.right:
            l.append(node.val)
    return l

#=======================test=================================

class Solution:
    def pathSum(self, root, targetSum: int) -> int:
        self.targetSum = targetSum
        self.n = 0
        que = deque([])
        def dfs_recursive(self,root,que):
            que.append(root.val)
            self.deal_que(que)
            if root.left:
                dfs_recursive(self,root.left,que.copy())
            if root.right:
                dfs_recursive(self,root.right,que.copy())
            return
        dfs_recursive(self,root,que)
        return self.n

    def deal_que(self,que):
        while sum(que)>self.targetSum:
            que.popleft()
        if sum(que)==self.targetSum:
            self.n += 1
            que.popleft()


s = Solution()
root = list2tree([1,2,3,4,5,6,7])
a = bfs(root)
print(a)
# root = list2tree([3,3,None,4,2])
