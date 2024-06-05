

def max_common_divisor(a,b):
    if a<b:
        a,b = b,a
    i = 1
    while i<b:
        if b%i==0 and a%(b//i)==0:
            return b//i
        i += 1
    return 0


a = max_common_divisor(2,1)
print(a)