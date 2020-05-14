import logging
import sys
import threading
import time
from itertools import combinations
from queue import Queue
from typing import List

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np

from opengl import WIDTH, HEIGHT, RANDOM, NUMBER_OF_PARTICLES, MOVEMENT_WORKER_THREADS, WINDOW_WIDTH, WINDOW_HEIGHT, \
    SCALE_X, SCALE_Y, RAY_CAST_WORKER_THREADS
from opengl.caster import Caster
from opengl.colour import Colour
from opengl.ray import Ray
from particle import Particle
from particle.mode import ColourMode
from particle.movement import Worker

_PARTICLES_LOSE_VELOCITY = False
_RAYCASTING = False

_zeroes = np.zeros((WIDTH, HEIGHT, 4), dtype=np.ubyte)
_frames = 0
_gravity_queue = Queue(maxsize=1)
_particles = []  # type: List[Particle]
_rays = []  # type: List[Ray]

colour_mode = ColourMode.SOLID

fps = 0
list_of_threads = []  # type: List[threading.Thread]
canvas = _zeroes.copy()
canvas_lock = threading.Lock()
drawn_event = threading.Event()


def set_colour_mode(mode: ColourMode):
    global colour_mode

    if mode == colour_mode:
        return

    colour_mode = mode
    for p in _particles:
        p.set_colour_mode(mode)

    logging.info("Set colour mode to " + str(mode))


def get_coord(position, bound):
    return (2 / bound) * position + -1


def get_fps(_):
    global _frames, fps

    fps = _frames
    _frames = 0
    glut.glutSetWindowTitle(f"Python Particles - {fps}fps")
    glut.glutTimerFunc(1000, get_fps, 0)


def render_string(text: str, x: int, y: int, colour: Colour = Colour(1, 1, 0)):
    gl.glColor3f(colour.red, colour.green, colour.blue)
    gl.glRasterPos2f(x, y)

    for i in range(len(text)):
        glut.glutBitmapCharacter(gl.OpenGL.GLUT.fonts.GLUT_BITMAP_HELVETICA_18, ord(text[i]))


def write_controls_text():
    render_string(text="g - Enable gravity",
                  x=get_coord(10, WIDTH),
                  y=get_coord(HEIGHT - 25, HEIGHT),
                  colour=Colour(1, 1, 1))


def display_callback():
    global _frames, _zeroes, canvas

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    if _RAYCASTING:
        canvas_lock.acquire(blocking=True)
        # print("drawing")

    image = canvas.copy()
    # canvas = np.zeros((WIDTH, HEIGHT, 4), dtype=np.ubyte)

    if not _RAYCASTING:
        for p in _particles:
            image[int(p.x)][int(p.y)] = p.colour.as_array()

    gl.glRasterPos2i(-1, -1)
    gl.glPixelZoom(SCALE_X, SCALE_Y)
    gl.glDrawPixels(WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)
    glut.glutSwapBuffers()

    _frames += 1
    # canvas = _zeroes.copy()

    if _RAYCASTING:
        drawn_event.set()
        canvas_lock.release()


def reshape_callback(w: int, h: int):
    gl.glClearColor(0, 0, 0, 0)
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)


def keyboard_callback(key, x, y):
    if key == b'\033' or key == b'q':
        sys.exit()
    elif key == b'g':
        _gravity_queue.put(1, block=False)

    elif key == b'1':
        set_colour_mode(ColourMode.SOLID)
    elif key == b'2':
        set_colour_mode(ColourMode.VELOCITY)
    elif key == b'3':
        set_colour_mode(ColourMode.DISTANCE)


def init_particles():
    for i in range(0, NUMBER_OF_PARTICLES):
        vr = 0.1 * np.sqrt(RANDOM.random()) + 0.05
        vphi = 2 * np.pi * RANDOM.random()
        vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)

        p = Particle(
            RANDOM.random() * WIDTH,
            RANDOM.random() * HEIGHT,
            vx,
            vy,
            Colour(RANDOM.random() * 255, RANDOM.random() * 255, RANDOM.random() * 255)
        )

        _particles.append(p)

    t = threading.Thread(target=move_particles, args=(), name="MovementThread")
    list_of_threads.append(t)
    t.start()

    if _RAYCASTING:
        t = threading.Thread(target=init_rays, args=(), name="CastingThread")
        list_of_threads.append(t)
        t.start()

    t = threading.Thread(target=listen_for_gravity, args=(), name="GravityThread")
    list_of_threads.append(t)
    t.start()

    if _PARTICLES_LOSE_VELOCITY:
        t = threading.Thread(target=loss_of_velocity, args=(), name="VelocityLossThread")
        list_of_threads.append(t)
        t.start()


def init_rays():
    global canvas

    for x in range(WIDTH):
        for y in range(HEIGHT):
            ray = Ray(np.array((255, 255, -255), dtype=np.float64),
                      np.array((x + 0.5, y + 0.5, 0), dtype=np.float64) - np.array((255, 255, -255), dtype=np.float64),
                      x,
                      y)
            _rays.append(ray)

    workers = []
    start_event = threading.Event()

    for i in range(1, RAY_CAST_WORKER_THREADS + 1):
        worker = Caster(i, start_event, _particles, _rays, canvas)
        workers.append(worker)
        worker.start()

    while True:
        drawn_event.wait()
        drawn_event.clear()
        canvas_lock.acquire(blocking=True)
        # with canvas_lock:
        # print("casting")
        start_event.set()
        for w in workers:
            w.finished()
        time.sleep(1)
        canvas_lock.release()


def handle_collisions(particle_combinations):
    for i, j in particle_combinations:
        if _particles[i].overlaps(_particles[j]):
            change_velocities(_particles[i], _particles[j])


def move_particles():
    workers = []
    start_event = threading.Event()
    particle_combinations = list(combinations(range(NUMBER_OF_PARTICLES), 2))

    for i in range(1, MOVEMENT_WORKER_THREADS + 1):
        worker = Worker(i, start_event, _particles)
        workers.append(worker)
        worker.start()

    while True:
        start_event.set()
        for w in workers:
            w.finished()
        # handle_collisions(particle_combinations)
        # time.sleep(0.01)


def loss_of_velocity():
    while True:
        for p in _particles:
            if p.vx != 0:
                p.vx = max((p.vx - 0.001), 0) if p.vx > 0 else min((p.vx + 0.001), 0)
            if p.vy != 0:
                p.vy = max((p.vy - 0.001), 0) if p.vy > 0 else min((p.vy + 0.001), 0)
        time.sleep(0.05)


def listen_for_gravity():
    logging.info("Waiting for gravity button event")
    while True:
        _ = _gravity_queue.get(block=True)

        for p in _particles:
            p.vx -= 0.1


def change_velocities(p1, p2):
    m1, m2 = p1.mass, p2.mass
    M = m1 + m2
    r1, r2 = p1.r, p2.r
    d = np.linalg.norm(r1 - r2) ** 2
    v1, v2 = p1.v, p2.v
    u1 = v1 - 2 * m2 / M * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
    u2 = v2 - 2 * m1 / M * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
    p1.v = u1
    p2.v = u2


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(threadName)s] | %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

    logging.info(f"Number of particles: {NUMBER_OF_PARTICLES}")
    logging.info(f"Particle movement workers: {MOVEMENT_WORKER_THREADS}")
    logging.info(f"Particles lose velocity over time: {_PARTICLES_LOSE_VELOCITY}")
    logging.info("Colour modes available:")

    for mode in ColourMode:
        logging.info(f"{ColourMode(mode).name} - {mode.value}")

    init_particles()

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glut.glutInitWindowPosition(100, 100)
    glut.glutCreateWindow('Particles')
    glut.glutDisplayFunc(display_callback)
    glut.glutIdleFunc(display_callback)
    glut.glutReshapeFunc(reshape_callback)
    glut.glutKeyboardFunc(keyboard_callback)
    glut.glutTimerFunc(1000, get_fps, 0)
    glut.glutMainLoop()
