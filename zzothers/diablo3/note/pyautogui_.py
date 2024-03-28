import time,random
import numpy as np

import keyboard
import win32api,win32con
import pyautogui
import pyperclip



def common(key:str):

    # mouse
    x, y = pyautogui.position()
    pyautogui.moveTo(0,0)
    pyautogui.keyDown(key)
    pyautogui.keyUp(key)
    pyautogui.click(x=0, y=0, clicks=1, interval=0.0, button='left')
    pyautogui.scroll(10)  # 向上滚动10个滚动单位

    # keyboard
    pyautogui.hotkey('ctrl', 'c') # 模拟按下并释放Ctrl+C组合键


    pyperclip.copy('text_to_copy')
    pyautogui.typewrite('Hello, world!') # 在当前焦点位置输入字符串
    pyautogui.alert('This is an alert!')

def color_():
    px = pyautogui.pixel(300,300)
    print(px)

def find_img(aim_path):
    try: 
        box = pyautogui.locateOnScreen(aim_path,grayscale=True,confidence=0.8)
        box_center = pyautogui.locateCenterOnScreen(aim_path,grayscale=True,confidence=0.8)
        pos = pyautogui.center(box)
        win32api.SetCursorPos(pos)
        print('find it: ',box)
        return box
    except Exception as e:
        if isinstance(e,pyautogui.ImageNotFoundException):
            print('not find anything')
        else:
            print('erro: ',e)

def click_img1(aim_path):
    box_center = pyautogui.locateCenterOnScreen(aim_path,grayscale=True,confidence=0.8)
    pyautogui.click(box_center)




    
if __name__=='__main__':
    # find_img('data/aim_img/aim.png')
    # click_img1('data/aim_img/aim.png')
    pass
    