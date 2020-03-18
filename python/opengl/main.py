import math
import sys
import threading
import time
from itertools import combinations
from typing import List

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np

from opengl import WIDTH, HEIGHT, RANDOM
from opengl.colour import Colour
from particle import Particle

_TRIANGLE_AMOUNT = 32
_TWICE_PI = 2.0 * math.pi

_particles = []  # type: List[Particle]
_NUMBER_OF_PARTICLES = 100

_zeroes = np.zeros((WIDTH, HEIGHT, 4), dtype=np.ubyte)
_frames = 0
fps = 0

list_of_threads = []  # type: List[threading.Thread]


def get_coord(position, bound):
    return (2 / bound) * position + -1


def get_fps(_):
    global _frames, fps

    fps = _frames
    _frames = 0
    glut.glutTimerFunc(1000, get_fps, 0)


def render_string(text: str, x: int, y: int):
    gl.glColor3f(1, 1, 0)
    gl.glRasterPos2f(x, y)

    for i in range(len(text)):
        glut.glutBitmapCharacter(gl.OpenGL.GLUT.fonts.GLUT_BITMAP_HELVETICA_18, ord(text[i]))


def display_callback():
    global _frames, _zeroes

    canvas = _zeroes.copy()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    for p in _particles:
        canvas[(int(p.x) - 1) % WIDTH][(int(p.y) - 1) % HEIGHT] = p.colour.as_array()

    gl.glRasterPos2i(-1, -1)
    gl.glDrawPixels(WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, canvas)
    render_string(str(fps), get_coord(10, WIDTH), get_coord(HEIGHT - 18, HEIGHT))
    glut.glutSwapBuffers()

    _frames += 1


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


def init_particles():
    for i in range(0, _NUMBER_OF_PARTICLES):
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

    t = threading.Thread(target=move_particles, args=())
    list_of_threads.append(t)
    t.start()


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


def handle_collisions():
    pairs = combinations(range(_NUMBER_OF_PARTICLES), 2)
    for i, j in pairs:
        if _particles[i].overlaps(_particles[j]):
            change_velocities(_particles[i], _particles[j])


def move_particles():
    while True:
        time.sleep(0.01)
        for p in _particles:
            p.move(7)
        handle_collisions()


if __name__ == "__main__":
    init_particles()

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(WIDTH, HEIGHT)
    glut.glutInitWindowPosition(100, 100)
    glut.glutCreateWindow('Example window')
    glut.glutDisplayFunc(display_callback)
    glut.glutIdleFunc(display_callback)
    glut.glutReshapeFunc(reshape_callback)
    glut.glutKeyboardFunc(keyboard_callback)
    glut.glutTimerFunc(1000, get_fps, 0)
    glut.glutMainLoop()

    for t in list_of_threads:
        t.join()
