from collections import deque,defaultdict
from typing import List

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord in wordList:
            target = wordList.index(endWord)
        else:
            return 0
        grouped_wordList = self.words_group(beginWord,wordList)
        graph = self.words2graph(grouped_wordList,len(wordList))
        graph.append([word_pair[0] for word_pair in grouped_wordList[0]])
        print(grouped_wordList,graph)
        return self.bfs_linked(len(wordList)-1,target,graph)        

    
    def linked(self,str1,str2):
        n_diff = 0
        for i in range(len(str1)):
            if str1[i]!=str2[i]:
                n_diff += 1
                if n_diff>1:
                    return False
        return True
    
    def word_distence(self,str1,str2):
        n_diff = 0
        for i in range(len(str1)):
            if str1[i]!=str2[i]:
                n_diff += 1
        return n_diff

    def bfs_linked(self,i,j,graph):
        visited = [0]*len(graph)
        shortest_path = 2
        bucket = deque([i])
        layer_bucket = deque()
        visited[i] = 1
        while bucket:
            curr = bucket.popleft()
            for n in graph[curr]:
                if not visited[n]:
                    layer_bucket.append(n)
                    visited[n] = 1
                    if n==j:
                        return shortest_path
            if not bucket:
                bucket = layer_bucket.copy()
                layer_bucket.clear()
                shortest_path += 1
        
        return 0

    def words_group(self,start,wordList: List[str]):
        grouped_wordList = [[] for _ in range(len(start))]
        for i,word in enumerate(wordList):
            grouped_wordList[self.word_distence(start,word)-1].append((i,word))
        return grouped_wordList

    def words2graph(self,grouped_wordList,len_):
        graph = [[] for i in range(len_+1)]
        for m in range(len(grouped_wordList)-1):
            temp_words = grouped_wordList[m]+grouped_wordList[m+1]
            for i in range(len(grouped_wordList[m])):
                for j in range(i+1,len(temp_words)):
                    if self.linked(temp_words[i][1],temp_words[j][1]):
                        graph[temp_words[i][0]].append(temp_words[j][0])
                        graph[temp_words[j][0]].append(temp_words[i][0])
        return graph


s = Solution()

beginWord ="hit"
endWord ="cog"
wordList =["hot","dot","dog","lot","log","cog"]

result = s.ladderLength(beginWord,endWord,wordList)
print(result)
