

def k_divide(nums,k):
    divide = [n*(len(nums)//k)+n-1 if n<=len(nums)%k else n*(len(nums)//k)+len(nums)%k-1 for n in range(1,k)]
    return divide