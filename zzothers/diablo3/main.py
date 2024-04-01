import time,random
import numpy as np

import keyboard
import win32api,win32con
import pyautogui
import pyperclip

def win32_click(x=-1,y=-1,button='left'):
        if x!=-1:
            win32api.SetCursorPos((x,y))
        if button=='left':
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
        elif button=='right':
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)


def mouse_pos(x=-1,y=-1):
    if x==-1:
        x, y = pyautogui.position()
    px = pyautogui.pixel(x,y)
    pyperclip.copy(f'{x,y,px}')

def mouse_xy_color(x=-1,y=-1):
    if x==-1:
        x, y = pyautogui.position()
    xs,ys = [],[]
    for i in range(-5,6):
        xs.append(pyautogui.pixel(x+i,y))
        ys.append(pyautogui.pixel(x,y+i))
    pyperclip.copy(f'{x,y}\n{xs}\n{ys}')

    # time.sleep(3)
    # xs1,ys1 = [],[]
    # for i in range(-5,6):
    #     xs.append(pyautogui.pixel(x+i,y))
    #     ys.append(pyautogui.pixel(x,y+i))
    # pyperclip.copy(f'{xs,xs1}')

def skill_collors():
    px1 = pyautogui.pixel(0,0)
    px2 = pyautogui.pixel(0,0)
    px3 = pyautogui.pixel(0,0)
    px4 = pyautogui.pixel(0,0)
    px5 = pyautogui.pixel(0,0)
    pyperclip.copy(f'{px1,px2,px3,px4,px5}')


def not_busy():
    if pyautogui.pixel(1140, 435)==(72, 64, 52) or pyautogui.pixel(1474, 103)==(165, 105, 41):
        return False
    return True
def auto_cast(a,b,c,d,e):
    notbusy = not_busy()
    if a==1:
        skilla_color = pyautogui.pixel(875, 1338)
        if 78<skilla_color[2]<88 and notbusy:
            keyboard.press_and_release('1')
    
    if b==1:
        skillb_color = pyautogui.pixel(965, 1338)
        if 78<skillb_color[2]<88 and notbusy:
            keyboard.press_and_release('2')
    elif b==2:
        buff_color = pyautogui.pixel(940, 1330)
        if buff_color[0]<20 and notbusy:
            keyboard.press_and_release('2')

    if c==1:
        skillc_color = pyautogui.pixel(1054, 1338)
        if 78<skillc_color[2]<88 and notbusy:
            keyboard.press_and_release('3')
    elif c==2:
        buff_color = pyautogui.pixel(1030, 1330)
        if buff_color[0]<20 and notbusy:
            keyboard.press_and_release('3')
    
    if d==1:
        skilld_color = pyautogui.pixel(1143,1338)
        if 78<skilld_color[2]<88 and notbusy:
            keyboard.press_and_release('4')
    elif d==2:
        buff_color = pyautogui.pixel(1118, 1330)
        if (buff_color[0]<20 or buff_color==(43, 44, 45)) and notbusy:
            keyboard.press_and_release('4')
    
    if e==1:
        skille_color = pyautogui.pixel(1236,1338)
        if 72<skille_color[2]<82 and notbusy:
            keyboard.press('e')
            win32_click()
            keyboard.release('e')
    elif e==2:
        buff_color = pyautogui.pixel(1211, 1330)
        if buff_color[0]<20 and notbusy:
            keyboard.press('e')
            win32_click()
            keyboard.release('e')
    
    bloode_color = pyautogui.pixel(66, 166)
    if bloode_color[0]<50:
        # print('bloode_color: ',bloode_color)
        keyboard.press_and_release('q')

def repeat(curr_time,interval):
    if curr_time[0]>=interval*2 and not_busy():
        curr_time[0] = 0
        # keyboard.press('e')
        # win32_click()
        # keyboard.release('e')
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
        
    else:
        curr_time[0] += 1

def buy_weapon():
    for _ in range(30):
        win32_click(button='right')
        

def decompose_legend():
    win32_click(214, 385)
    for i in range(3):
        y = 777+i*134
        for j in range(10):
            x = 1906+j*67
            color1 = pyautogui.pixel(x+10,y+10)
            color2 = pyautogui.pixel(x-10,y+10)
            color3 = pyautogui.pixel(x-10,y-10)
            color4 = pyautogui.pixel(x+10,y-10)
            cond_a = color1[0]>16 or color1[1]>15 or color1[2]>10
            cond_b = color2[0]>15 or color2[1]>15 or color2[2]>10
            cond_c = color3[0]>18 or color3[1]>15 or color3[2]>10
            cond_d = color4[0]>18 or color4[1]>17 or color4[2]>10
            if cond_a + cond_b + cond_c + cond_d>2:
                time.sleep(0.04)
                win32_click(x, y)
                time.sleep(0.02)
                # win32_click(1137, 496)
                keyboard.press_and_release('enter')
    # win32_click(button='right')
                
def decompose_weapon():
    if pyautogui.pixel(517, 373)==(64, 54, 13):
        win32_click(514,388)
        keyboard.press_and_release('enter')
        time.sleep(0.1)
    if pyautogui.pixel(422, 372)==(11, 36, 69):
        win32_click(422, 372)
        keyboard.press_and_release('enter')
        time.sleep(0.1)
    if pyautogui.pixel(344, 373)==(50, 53, 50):
        win32_click(344, 373)
        keyboard.press_and_release('enter')
        time.sleep(0.2)
    decompose_legend()

def start_auto_cast():
    curr_time = [0]
    keyboard.press('1')
    while keyboard.is_pressed('f3')==False:
        auto_cast(0,1,1,1,1)
        repeat(curr_time,3)
        time.sleep(0.5)
    keyboard.release('1')
def main():
    keyboard.add_hotkey('f2', start_auto_cast)
    keyboard.add_hotkey('ctrl+1', buy_weapon)
    keyboard.add_hotkey('ctrl+e', decompose_weapon)
    keyboard.add_hotkey('f4', lambda: mouse_pos(1118, 1330))
    keyboard.add_hotkey('f8', lambda: mouse_xy_color())
    # keyboard.add_hotkey('ctrl+u', lambda: trigger_func_param('ctrl+u'))
    keyboard.wait('f12')


if __name__=='__main__':
    main()

