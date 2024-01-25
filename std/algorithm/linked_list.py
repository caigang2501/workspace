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

def reverseBetween(head, left: int, right: int):
    tail,prev,curr = head,head,head
    i = 1
    if left==right:
        return head
    elif left==1:
        curr,next = head.next,head.next
        tail.next = None
        i = 2
    while i<=right:
        if i < left-1:
            curr = curr.next
        elif i == left-1:
            prev = curr
            tail = curr
            curr = curr.next
            next = curr
        else:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        i += 1
    if tail.next==None:
        tail.next = curr
    else:
        tail.next.next = curr
        tail.next = prev
        return head
    return prev

l = LinkedList([1,2])
head = l.head
a = reverseBetween(head,1,2)
display(a)
