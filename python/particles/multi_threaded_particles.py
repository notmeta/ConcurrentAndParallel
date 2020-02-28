import random
import threading

_MAX_VALUE = 50
_MIN_VALUE = 0


def thread_main(thread_id, particles):
    for x in range(100_000):
        move(particles[thread_id])


def main():
    particles = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [6, 6],
        [6, 6],
    ]

    print_particles(particles)

    list_of_threads = []
    num_threads = len(particles)

    for id in range(num_threads):
        t = threading.Thread(target=thread_main, args=(id, particles))
        list_of_threads.append(t)
        t.start()

    print("shuffling particles")

    for t in list_of_threads:
        t.join()

    print("finished shuffling particles")

    print_particles(particles)


def rand() -> int:
    return int((random.random() * 10) - 5)


def move(particle):
    particle[0] = clip(particle[0] + rand())
    particle[1] = clip(particle[1] + rand())


def clip(current: int, max_value: int = _MAX_VALUE, min_value: int = _MIN_VALUE) -> int:
    return max(min(current, max_value), min_value)


def print_particles(particles):
    for p in particles:
        print(p)


if __name__ == '__main__':
    main()
