import threading
import time
threads = []

print("hello")

def doWork(i):
    print("i = ",i)
    time.sleep(5)

for i in range(1,4):
    thread = threading.Thread(target=doWork, args=(i,))
    threads.append(thread)
    thread.start()

# you need to wait for the threads to finish
for thread in threads:
    thread.join()

print("Finished")
