import pyautogui




def test():
    im1 = pyautogui.screenshot()
    im1.save('my_screenshot.png')
    # im2 = pyautogui.screenshot('my_screenshot2.png')



if __name__=='__main__':
    test()