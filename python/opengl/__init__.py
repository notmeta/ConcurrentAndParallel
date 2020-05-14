import random

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

SCALE_X = 2
SCALE_Y = 2

WIDTH = int(WINDOW_WIDTH / SCALE_X)
HEIGHT = int(WINDOW_HEIGHT / SCALE_Y)

RANDOM = random.SystemRandom()
NUMBER_OF_PARTICLES = 100
MOVEMENT_WORKER_THREADS = 5
RAY_CAST_WORKER_THREADS = 5
PARTICLE_COLLISIONS = True
