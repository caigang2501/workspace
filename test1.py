from collections import deque

def singleNumber(nums) -> int:
    a = sum(nums)
    b = 1
    while a%3==0:
        nums = [num//3 for num in nums if num % 3 == 0]
        b = b*3
        a = sum(nums)
    for num in nums:
        if (a-num)%3==0 and (a-3*num)%3!=0:
            print(a,b)
            return num*b

a = [2,2,2,-1,-1,-1,8,-7,0,-7,0,-7,0]

print(singleNumber(a))