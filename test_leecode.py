
from typing import List,Optional
from collections import deque

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        board.reverse()
        for n,row in enumerate(board):
            if n%2==1:
                row.reverse()
        board = [elem for row in board for elem in row]

        board[0] = 0
        players = [0]
        step = 1
        new_players = []
        while players:

            alive = [1]*len(players)
            players.sort()
            players.reverse()
            for a,player in enumerate(players):
                t = 0
                for i in range(1,7):
                    if player+i==len(board)-1:
                        return step
                    
                    if board[player+i]==-1:
                        t = i
                    elif board[player+i]>0:
                        if board[player+i]==len(board):
                            return step
                        new_players.append(board[player+i]-1)
                    board[player+i] = 0
                if t==0:
                    alive[a] = 0
                else:
                    players[a] = player+t
            players = [player for a,player in enumerate(players) if alive[a]==1]
            for new_player in new_players:
                players.append(new_player)
                if board[new_player]==-1:
                    board[new_player] = 0
            new_players = []
            step += 1
        return -1

        

s = Solution()
l = [[-1,1,1,1],[-1,7,1,1],[16,1,1,1],[-1,1,9,1]]
result = s.snakesAndLadders(l)
print(result)
