import sys

class Node:
    def __init__(self, data=0):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self,lst=[]):
        self.head = None
        for e in lst:
            self.append(e)

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        if self.head is None:
            return

        if self.head.data == data:
            self.head = self.head.next
            return

        current_node = self.head
        while current_node.next and current_node.next.data != data:
            current_node = current_node.next

        if current_node.next:
            current_node.next = current_node.next.next

    def display(self):
        current_node = self.head
        while current_node:
            print(current_node.data, end=" -> ")
            current_node = current_node.next
        print("None")

    def reverseList(self, head):
        prev = None
        curr = head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev

#============================test===========================
def display(current_node):
    while current_node:
        print(current_node.data, end=" -> ")
        current_node = current_node.next
    print("None")

class Solution:
    def mergeKLists(self, lists):
        head = Node()
        t = head
        while len(lists)>1:
            i = 0
            mini,minh = 0,lists[0]
            while i < len(lists):
                if lists[i].data<minh.data:
                    mini,minh = i,lists[i]
                i += 1
            t.next = minh
            t = t.next
            if minh.next:
                lists[mini] = minh.next
            else:
                lists.pop(mini)
        t.next = lists[0]
        return head.next

l = LinkedList([1,4,5])
# [[1,4,5],[1,3,4],[2,6]]
lst = [LinkedList([1,4,5]).head,LinkedList([1,3,4]).head,LinkedList([2,6]).head]
s = Solution()
a = s.mergeKLists(lst)
display(a)
