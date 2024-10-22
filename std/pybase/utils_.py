import os,datetime

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

if __name__=='__main__':
    date_test()
