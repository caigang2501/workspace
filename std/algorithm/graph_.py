
# 并查集
def disjoint_set(edges):
    parts = []
    for edge in edges:
        temp = []
        for i in range(len(parts)):
            for x in edge:
                if x in parts[i]:
                    temp.append([i,x])
        if len(temp)==1:
            edge.remove(temp[0][1])
            parts[temp[0][0]] += edge
        elif len(temp)==2:
            if temp[0][0] != temp[1][0]:
                parts[temp[0][0]] += parts[temp[1][0]]
                parts.pop(temp[1][0])
        else:
            parts.append(edge)
    return parts

print(disjoint_set([[6,7],[1,2],[4,5],[2,3],[5,6]]))

