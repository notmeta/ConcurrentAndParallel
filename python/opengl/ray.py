import numpy as np


class Ray:
    def __init__(self, a, b, x, y):
        self.A = a
        self.B = b
        self.X = x
        self.Y = y

    def origin(self):
        return self.A

    def direction(self):
        return self.B

    def point_at_parameter(self, t: float):
        return np.add(self.A, np.cross(self.B, t))

    def does_hit_sphere(self, center, radius):
        oc = self.origin() - center
        a = np.dot(self.direction(), self.direction())
        b = 2 * np.dot(oc, self.direction())
        c = np.dot(oc, oc) - radius * radius
        discriminant = b * b - 4 * a * c
        return discriminant >= 0
