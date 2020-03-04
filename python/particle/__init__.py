import random

from opengl import WIDTH, HEIGHT
from opengl.colour import Colour


class Particle(object):

    def __init__(self, x: int, y: int, colour: Colour):
        self.x = x
        self.y = y
        self.colour = colour

    def move(self):
        modifier = 8

        self.x += int(random.random() * modifier - (modifier / 2))
        self.y += int(random.random() * modifier - (modifier / 2))

        self.x = max(min(WIDTH, self.x), 0)
        self.y = max(min(HEIGHT, self.y), 0)
