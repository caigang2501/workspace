import multiprocessing as mp
import time


def worker(num,works=20):
    print(f'========================Worker {num} start working==========================')
    """简单的工作函数，打印工作编号"""
    for i in range(works):
        time.sleep(0.1)
        if i>15:
            print(f'Worker {num} is working {i}')
    print(f'------------------------Worker {num} finished working-------------------------')
    return num

def demo1():
    # 创建进程池，指定进程数量为3
    with mp.Pool(processes=2) as pool:
        # 使用apply_async方法异步地将任务提交给进程池中的工作进程执行
        result0 = pool.apply_async(worker, (0, 20))
        result1 = pool.apply_async(worker, (1, 20))
        result2 = pool.apply_async(worker, (2, 20))
        # 获取每个工作进程的返回结果
        results = [result0.get(),result1.get(),result2.get()]

    print(results)

def demo2():
    with mp.Pool(processes=3) as pool:
        results = pool.map(worker, [i for i in range(6)])
        print(results)

def demo3():
    with mp.Pool(processes=3) as pool:
        results = pool.starmap(worker, [(i, 20) for i in range(6)])
        print(results)

def demo4_sub(shared_value):
    # 在进程中递增共享的整数值
    with shared_value.get_lock():
        shared_value.value += 1

def demo4():
    shared_value = mp.Value('i', 0)
    p = mp.Process(target=demo4_sub,args=(shared_value,))
    p.start()
    p.join()

    print("Final shared value:", shared_value.value)

def demo5_sub(m,dic):
    with mp.Pool(processes=3) as pool:
        results = pool.map(worker, [m+i for i in range(6)])
        dic[m] = results
def demo5():
    with mp.Manager() as mg:
        mg_dic = mg.dict()
        p0 = mp.Process(target=demo5_sub,args=(10,mg_dic))
        p1 = mp.Process(target=demo5_sub,args=(20,mg_dic))
        p0.start(),p1.start()
        p0.join(),p1.join()
        print(mg_dic)

if __name__=='__main__':

    demo5() 



