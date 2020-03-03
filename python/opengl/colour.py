from __future__ import annotations


class Colour(object):

    def __init__(self, red: float, green: float, blue: float):
        self.red = red
        self.green = green
        self.blue = blue

    @classmethod
    def from_int(cls, red: int, green: int, blue: int) -> Colour:
        return Colour(red / 255, green / 255, blue / 255)
