import random


def main():
    particles = [
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6]
    ]

    print_particles(particles)

    print("shuffling particles")

    for x in range(2_500_000):
        for p in particles:
            move(p)

    print("finished shuffling particles")

    print_particles(particles)


def move(particle):
    particle[0] += (random.random() * 10) - 5
    particle[1] += (random.random() * 10) - 5


def print_particles(particles):
    for p in particles:
        print(p)


if __name__ == '__main__':
    main()
