import random

from opengl import WIDTH, HEIGHT
from opengl.colour import Colour


class Particle(object):

    def __init__(self, x: int, y: int, colour: Colour):
        self.x = x
        self.y = y
        self.colour = colour
        self.radius = 0.013

        self.triangles = int(self.radius * 4000) + 1

    def move(self):
        modifier = 4

        self.x += (random.random() * modifier - (modifier / 2))
        self.y += (random.random() * modifier - (modifier / 2))

        # print(self.x)

        self.x = max(min(WIDTH, self.x), 0)
        self.y = max(min(HEIGHT, self.y), 0)
