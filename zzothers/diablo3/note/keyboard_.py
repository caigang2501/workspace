
import keyboard

def common():
    keyboard.is_pressed('q')
    

def trigger_func():
    print("triggered")

def trigger_func_param(param):
    print("triggered: ",param)

def add_hotkey():
    keyboard.add_hotkey('f4', trigger_func)
    keyboard.add_hotkey('ctrl+u', lambda: trigger_func_param('ctrl+u'))
    keyboard.wait('esc')

def on_key_pressed(event):
    if event.name == 'q':
        print("按下了 'q' 键！")

def add_key():
    keyboard.on_press(on_key_pressed)
    keyboard.wait('esc')

if __name__=='__main__':
    # add_hotkey()
    add_key()
    
    