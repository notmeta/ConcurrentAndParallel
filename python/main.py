import logging
import threading
import time

lock = threading.RLock()


def thread_function(id: int):
    try:
        logging.info("Thread %d: Begin", id)
        duration = 1
        time.sleep(duration)

        with lock:
            print('Thread', id, 'sleeping for', duration, 'secs')
            raise Exception("fake exception")

        # lock.release()
        logging.info("Thread %d: End", id)

    except:
        logging.info("Thread %d: Exception", id)


def main():
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main : Begin")

    list_of_threads = []
    num_threads = 3

    logging.info("Main : Creating and starting %d threads", num_threads)

    for id in range(num_threads):
        t = threading.Thread(target=thread_function, args=(id,))
        list_of_threads.append(t)
        t.start()

    logging.info("Main : Waiting for threads")
    for t in list_of_threads:
        t.join()

    logging.info("Main: End")


if __name__ == '__main__':
    main()
