import numpy as np
import random,sys,os
sys.path.append(os.getcwd())
from std.mymath.statistic_tools import plot_distribution
class Hanoi_tower:
    def __init__(self,level:int) -> None:
        self.level = level
        self.l1 = [level-i for i in range(level)]
        self.l2 = []
        self.l3 = []
        self.ls = [self.l1,self.l2,self.l3]

    def moveable(self,start,aim):
        if 0<=start<3 and 0<=aim<3:
            start = self.ls[start]
            aim = self.ls[aim]
        else:
            return False
        
        can_move = False
        if len(start)!=0:
            if len(aim)==0:
                can_move = True
            elif start[-1]<aim[-1]:
                can_move = True
        
        return can_move
    
    def move(self,action):
        start,aim = action
        if 0<=start<3 and 0<=aim<3:
            start = self.ls[start]
            aim = self.ls[aim]
        else:
            return 
        aim.append(start.pop())

    def get_available_actions(self):
        available_actions = []
        for i in range(3):
            if len(self.ls[i])>0:
                for j in range(3):
                    if i!=j and self.moveable(i,j):
                        available_actions.append((i,j))
        return available_actions

def random_finish_time(ht:Hanoi_tower):
    for i in range(50000):
        available_actions = ht.get_available_actions()
        ht.move(random.choice(available_actions))
        if len(ht.l3)==ht.level:
            return i


if __name__=='__main__':
    ht = Hanoi_tower(4)
    plot_distribution(random_finish_time,ht,num_samples=10000)
      