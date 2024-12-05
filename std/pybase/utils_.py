import os,datetime,shutil

def file_test():
    with open('example.txt', 'w') as file:
        file.write('qwer')

    with open('example.txt', 'a') as file:
        file.write('asdf')

    with open('example.txt', 'r') as file:
        content = file.read()
        print(content)

def date_test():
    d = datetime.datetime.now().date()
    t = datetime.datetime.now().time()
    print(d,type(d))

def clear_dir(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            clear_dir(file_path)
    # shutil.rmtree(os.getcwd()+'/path')

if __name__=='__main__':
    date_test()
