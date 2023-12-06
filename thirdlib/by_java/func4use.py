import sys

def funa(a,b):
    return a + b

if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    result = funa(arg1,arg2)

    print(result)
