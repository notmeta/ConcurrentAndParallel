import math
import random
import sys
from typing import List

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from opengl import WIDTH, HEIGHT
from opengl.colour import Colour
from particle import Particle

_TRIANGLE_AMOUNT = 32
_TWICE_PI = 2.0 * math.pi

_particles = []  # type: List[Particle]
_NUMBER_OF_PARTICLES = 250


def get_coord(position, bound):
    return (2 / bound) * position + -1


def draw_circle(x: float, y: float, radius: float, triangles: int, colour: Colour):
    x = get_coord(x, WIDTH)
    y = get_coord(y, HEIGHT)

    # gl.glColor3f(colour.red, colour.green, colour.blue)
    gl.glColor3f(x, y, abs(x - y))
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glVertex3f(x, y, 0)
    for i in range(0, triangles + 1):
        gl.glVertex2f(
            x + ((0 + radius) * math.cos(i * _TWICE_PI / triangles)),
            y + ((0 - radius) * math.sin(i * _TWICE_PI / triangles))
        )
    gl.glEnd()


def display_callback():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()

    for p in _particles:
        draw_circle(p.x, p.y, p.radius, p.triangles, p.colour)
        # if random.random() >= 0.1:
        p.move()

    # draw_circle(0, 0, 0.01, Colour.from_int(100, 100, 190))
    # draw_circle(WIDTH, HEIGHT, 0.01, Colour.from_int(100, 100, 190))
    # draw_circle(WIDTH / 2, HEIGHT / 2, 0.01, Colour.from_int(100, 100, 190))

    # time.sleep(0.008)

    glut.glutSwapBuffers()


def reshape_callback(w: int, h: int):
    # gl.glClearColor(1, 1, 1, 1)
    gl.glClearColor(0, 0, 0, 0)
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)


def keyboard_callback(key, x, y):
    if key == b'\033':
        sys.exit()
    elif key == b'q':
        sys.exit()


def init_particles():
    for i in range(0, _NUMBER_OF_PARTICLES):
        p = Particle(WIDTH / 2, HEIGHT / 2, Colour(random.random(), random.random(), random.random()))

        p.radius = random.random() / 32

        _particles.append(p)


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
    glut.glutMainLoop()
