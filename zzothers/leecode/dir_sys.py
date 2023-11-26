class DirNode:
    def __init__(self,last,name):
        self.val = {}
        self.name = name
        self.last = last

s = ['mkdir a','mkdir b','cd a','mkdir c','mkdir d','cd ..','cd b','mkdir e','mkdir f','cd ..','cd a','cd d','pwd']
head = DirNode(None,'')
node = head

path = ''
for line in s:
    a = line.split()
    if a[0] == 'mkdir':
        if a[1] not in node.val:
            node.val[a[1]] = DirNode(node,a[1])
    elif a[0] == 'cd':
        if a[1]== '..':
            node = node.last
        elif a[1] in node.val:
            node = node.val[a[1]]
    else:
        while node.last:
            path = node.name + '/' + path
            node = node.last
        path = '/' + path
print(path)