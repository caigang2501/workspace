# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:28:38 2021

@author: dell
"""
import math

class Solution:  
                
    def nCombination(self,l:list,n:int) -> list:
        def combin(nums:list,temp:list,idx:int):
            while idx < len(nums):
                if len(temp)<n-1:
                    combin(nums,temp+[nums[idx]],idx+1)
                else:
                    ans.append(temp+[nums[idx]])
                idx += 1
        ans = []
        combin(l,[],0)
        return ans
    
    #获取n排列的第k个排列 
    def getPermutation(self, n: int, k: int) -> str:
        i = 1
        l = []
        ans = list(range(n))
        ans.append(n)
        ans.remove(0)
        
        while math.factorial(i)<k:
            i += 1
            l.append(ans.pop())
            
        l.append(ans.pop())
        l.reverse()
        while i>0:
            t = math.ceil(k/math.factorial(i-1))
            k = k % math.factorial(i-1)
            ans.append(l.pop(t-1))                 
            i -= 1
                
        s = ""
        for x in ans:
            s += "%x" % (x)
        return s
    
    # 二分查找
    def divsearch(self,nums:list,target:int) -> bool:
        i = 0
        j = len(nums)-1
        if nums[0]==target or nums[-1]==target:
            return True
        while j-i>1:
            if target<nums[(j-i)//2+i]:
                j = (j-i)//2+i
            elif nums[(j-i)//2+i]<target:
                i = (j-i)//2+i
            else:
                return True
        return False
    
    def divsearch_pst(self,nums:list,target:int):
        i = 0
        j = len(nums)-1
        if target<nums[0]:
            return (-1,0)
        elif nums[-1]<target:
            return (j,j+1)
        while j-i>1:
            if target<nums[(j-i)//2+i]:
                j = (j-i)//2+i
            elif nums[(j-i)//2+i]<target:
                i = (j-i)//2+i
            else:
                return ((j-i)//2+i,(j-i)//2+i)
        return (i,j)

a = [1,2,3,4,5]
s = Solution()
ans = s.getPermutation(15,100000000000)
print(ans,math.factorial(15))





