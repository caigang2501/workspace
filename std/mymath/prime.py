import math

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


def countPrimes(n: int) -> int:
    i = 3
    ans = [2]
    while i < n:
        if isPrim(i):
            ans.append(i)
        i += 2
    return ans

def test(n: int) -> int:
    i = 3
    ans = 1/2
    while i < n:
        if isPrim(i):
            ans += 1/i
        i += 2
    return ans

def test1(n: int) -> int:
    i = 3
    ans = 1/2
    c = 1/2
    while i < n:
        if isPrim(i):
            c = c/i
            ans += c
        i += 2
    
    return (ans,c)


if __name__=='__main__':
    result = test1(1000000)
    print(result[0],1-result[0])
    