from multiprocessing import Process

def fun1():
    print("good")


def test():
    p = Process(target=fun1,args=())
    p.start()
    p.join()

if __name__=="__main__":
    test()