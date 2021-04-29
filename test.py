from multiprocessing import Process, Queue


def f1(out: list):
    for i in range(100):
        print(i)
    out.append(range(10))


def f1(out: list):
    for i in range(50):
        print(i)
    out.append(range(5))


if __name__ == '__main__':
    shared_q = Queue()
    p = Process(target=f1, args=(shared_q,))
    p.st
