
import keyboard

from module1 import constent
from module1.utils import change
from flask import g,session


def test1():
    change()
    a = [[constent.param],constent.lst]
    print(a)
    return a
    
if __name__=='__main__':
    # keyboard.add_hotkey('ctrl+d',print,('asdf',))
    # keyboard.add_hotkey('d',test1)
    # keyboard.wait('esc')
    test1()

