import threading
import time
import multiprocessing

class myThread (threading.Thread):
    def __init__(self, threadID, name, delay,counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.delay = delay
        self.counter = counter
    def run(self):
        print ("开始线程：" + self.name)
        print_time(self.name, self.delay, self.counter)
        print ("退出线程：" + self.name)

def print_time(threadName, delay, counter):
    while counter:
        # if exitFlag:
        #     threadName.exit()
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

def testt():
    thread1 = myThread(1, "Thread-1", 1,5)
    thread2 = myThread(2, "Thread-2", 1,1)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    print ("退出主线程")


if __name__ == "__main__":
    testt()




