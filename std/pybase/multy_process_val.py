import multiprocessing as mp
import time


def worker_queue(num,works,result_queue):
    print(f'========================Worker {num} start working==========================')
    for i in range(works):
        time.sleep(0.1)
        if i>15:
            print(f'Worker {num} is working {i}')
    print(f'------------------------Worker {num} finished working-------------------------')
    result_queue.put(works)
    return num
def demo4():
    result_queue0 = mp.Queue()
    process0 = mp.Process(target=worker_queue, args=(0,20,result_queue0))
    process0.start()
    process0.join()
    print(result_queue0.get())

def f0(n, a):
    n.value = 2.71828
    for i in range(len(a)):
        a[i] = -a[i]
def f1(n, a):
    n.value = 3.1415927
    for i in range(len(a)-5):
        a[i] = -a[i]
def demo5():
    num = mp.Value('d', 0.0)
    arr = ('i', range(10))

    p0 = mp.Process(target=f0, args=(num, arr))
    p1 = mp.Process(target=f1, args=(num, arr))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print(num.value)
    print(arr[:])


if __name__=='__main__':
    demo4()



