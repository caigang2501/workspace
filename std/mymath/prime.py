import math

def countPrimes(n: int) -> int:
    def isPrim(n:int) -> bool:
        t = math.ceil(math.sqrt(n))
        if n%2 == 0:
            return False
        else:
            i = 3
            while i <= t:
                if n%i == 0:
                    return False
                i += 2
            return True
        
    if n < 3:
        return 0
    
    i = 3
    ans = []
    while i < n:
        if isPrim(i):
            ans.append(i)
        i += 2
    return ans

if __name__=='__main__':
    result = countPrimes(1000)
    print(result,len(result))
    