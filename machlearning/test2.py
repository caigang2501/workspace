g = 1
def test(n):
    m = 2
    g = 4
    print(g)
    def subtest():
        m = 9
        n = 9
    print(n,m)
    subtest()

test(3)
print(g)