import numpy as np

from opengl import WIDTH, HEIGHT
from opengl.colour import Colour


class Particle(object):

    def __init__(self, x: int, y: int, vx: int, vy: int, colour: Colour):
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = 1
        self.mass = self.radius ** 2

        self.colour = colour

    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def move(self, dt: float):
        # self.r += self.v * dt
        np.add(self.r, self.v * dt, out=self.r, casting="unsafe")

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > HEIGHT:
            self.y = HEIGHT - self.radius
            self.vy = -self.vy
