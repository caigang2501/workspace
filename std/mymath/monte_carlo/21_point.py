import random



def play4banker(pokers,point):
    sum = 0
    life = 1
    while True:
        card = pokers.pop()
        if card == 11:
            life += 1
        sum += card
        if sum>21:
            life -= 1
            if life == 0:
                return -1
        elif sum>point:
            return 1
        elif sum==point:
            return 0
        
def play4player(pokers,point):
    sum = 0
    life = 1
    while True:
        card = pokers.pop()
        if card == 11:
            life += 1
        sum += card
        if sum>21:
            life -= 1
            if life == 0:
                return -1
        elif sum>=point:
            point = sum
            break
    return -play4banker(pokers,point)

def test_banker():
    point = 18
    repeat = 500000

    win = 0
    for i in range(repeat):
        pokers = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
        random.shuffle(pokers)
        win += play4banker(pokers,point)
    
    print(point,':',round(win*100/repeat,4))

def test_player():
    point = 14
    repeat = 500000

    win = 0
    for i in range(repeat):
        pokers = [11,2,3,4,5,6,7,8,9,10,10,10,10]*4
        random.shuffle(pokers)
        win += play4player(pokers,point)
    
    print(point,':',round(win*100/repeat,4))

if __name__=='__main__':
    test_player()

    