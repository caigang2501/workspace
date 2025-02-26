import threading,time,queue

condition = threading.Condition()
count = 1

def print_odd():
    global count
    for _ in range(5):
        with condition:
            while count % 2 == 0:
                condition.wait()
            print(f"奇数线程: {count}")
            count += 1
            condition.notify_all()

def print_even():
    global count
    for _ in range(5):
        with condition:
            while count % 2 == 1:
                condition.wait()
            print(f"偶数线程: {count}")
            count += 1
            condition.notify_all()

def turn_print():
    t1 = threading.Thread(target=print_odd)
    t2 = threading.Thread(target=print_even)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


q = queue.Queue()

def producer():
    for i in range(5):
        with condition:
            q.put(i)
            print(f"生产者: 生产 {i}")
            condition.notify_all()  # 唤醒消费者
        time.sleep(1)

def consumer():
    for _ in range(5):
        with condition:
            while q.empty():
                condition.wait()  # 等待生产者通知
            item = q.get()
            print(f"消费者: 消费 {item}")

def producer_consumer():
    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__=='__main__':
    turn_print()
    producer_consumer()

