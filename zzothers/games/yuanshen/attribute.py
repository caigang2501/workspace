

def test():
    r2 = round(2.5*5,1)
    for i in range(1,19):
        a = 25+round(9.2*i,1)
        a1 = 25+round(9.2*(19-i),1)

        b = 3.1*(i)
        r1 = round((42+b)*(85+2*b)/100,1)
        print(i,a1,round(9.2*100/(a+76),1),' ',round((42+b),1),round((85+2*b),1),' ',r1,(round((r1-r2)*100/(100+r2),1)),round((100+a1+30)*(100+r1)/100,1))
        r2 = r1

if __name__=='__main__':
    test()