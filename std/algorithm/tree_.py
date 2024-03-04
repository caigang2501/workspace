from collections import deque
from queue import Queue
from typing import List,Optional


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


def bfs_by_layer(root):
    if not root:
        return []

    result = []
    lst_t = []
    queue = deque([root])
    dqt = deque()

    while queue:
        current = queue.popleft()
        lst_t.append(current.val)

        if current.left:
            dqt.append(current.left)
        if current.right:
            dqt.append(current.right)
        
        if not queue:
            queue = dqt.copy()
            dqt.clear()
            result.append(lst_t.copy())
            lst_t.clear()

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

path = []
def find_val(root:TreeNode,target:int,ancestors:List=[]):
    ancestors.append(root.val)
    global path
    if root.val==target:
        path = ancestors.copy()
    if root.left and len(path)==0:
        find_val(root.left,target,ancestors)
    if root.right and len(path)==0:
        find_val(root.right,target,ancestors)
    ancestors.pop()
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
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.path = []
        self.find = False
        self.ancestors = []
        self.p,self.q = p,q
        self.root = root
        self.node = root
        return self.find_val(root)

    def find_val(self,root:TreeNode):
        self.ancestors.append(root.val)
        if root.val==self.p or root.val==self.q:
            if len(self.path)==0:
                self.path = self.ancestors.copy()
            else:
                self.node = self.get_node(self.root)
                self.find = True
        if root.left and not self.find:
            self.find_val(root.left)
        if root.right and not self.find:
            self.find_val(root.right)
        self.ancestors.pop()
        return self.node
    
    def get_node(self,node: 'TreeNode'):
        print(self.path)
        print(self.ancestors)
        i = 1
        len_min = min(len(self.path),len(self.ancestors))
        while i<len_min and self.path[i]==self.ancestors[i]:
            if node.left and node.left.val==self.path[i]:
                node = node.left
            else:
                node = node.right
            i += 1
        return node
s = Solution()
root = list2tree([3,5,1,6,2,0,8,None,None,7,4])
# result = s.lowestCommonAncestor(root,6,4)
print(bfs_by_layer(root))
# root = list2tree([3,3,None,4,2])

# 1 2 3 4 5 6 7  bfs
# 1 2 4 5 3 6 7  dfs
# 1 2 4 5 3 6 7  先
# 4 2 5 1 6 3 7  中
# 4 5 2 6 7 3 1  后
