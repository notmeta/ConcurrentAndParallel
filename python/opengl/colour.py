from __future__ import annotations

from typing import List


class Colour(object):

    def __init__(self, red: int, green: int, blue: int):
        self.red = int(red)
        self.green = int(green)
        self.blue = int(blue)

    def as_array(self) -> List[int]:
        return [self.red, self.green, self.blue, 0]  # 0 for alpha
