from collections import deque
from queue import Queue


class TreeNode:
    def __init__(self, val,next=None):
        self.val = val
        self.left = None
        self.right = None
        self.next = next

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
    def connect(self, root):
        if not root:
            return []

        curr = root
        while curr:
            curr.next = 1
            curr = curr.right
        curr = root
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if curr.next==1:
                curr.next = None
            else:
                curr.next = current
            curr = current
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

        return root
s = Solution()
root = list2tree([1,2,3,4,5,None,7])
a = s.connect(root)
print(a)
# root = list2tree([3,3,None,4,2])
