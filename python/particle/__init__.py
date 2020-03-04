import math
import random

import numpy as np

from opengl import WIDTH, HEIGHT
from opengl.colour import Colour


class Particle(object):

    def __init__(self, x: int, y: int, colour: Colour):
        self.x = x
        self.y = y
        self.colour = colour

        self._x_path = list(self._brownian_path(int(random.random() * WIDTH) + 1))
        self._y_path = list(self._brownian_path(int(random.random() * HEIGHT) + 1))
        self._x_move_counter = 1
        self._y_move_counter = 1

    def move(self):
        self.x += self._x_path[self._x_move_counter - 1]
        self.y += self._y_path[self._y_move_counter - 1]

        if self._x_move_counter >= len(self._x_path):
            self._x_path = self._brownian_path(WIDTH)
            self._x_move_counter = 0

        if self._y_move_counter >= len(self._y_path):
            self._y_path = self._brownian_path(HEIGHT)
            self._y_move_counter = 0

        self.x %= WIDTH
        self.y %= HEIGHT

        self._x_move_counter += 1
        self._y_move_counter += 1

    @staticmethod
    def _brownian_path(coord: int) -> int:
        t_sqrt = math.sqrt(1 / coord)
        z = np.random.randn(coord)
        z[0] = 0
        b = np.cumsum(t_sqrt * z)

        return b
