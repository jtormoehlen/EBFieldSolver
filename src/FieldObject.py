import numpy as np
from matplotlib import pyplot as plt


# (*$a^2 + b^2 = c^2$*)
class PointCharge:
    epsilon_0 = 8.85E-12
    const = 1 / (4 * np.pi * epsilon_0)

    def __init__(self, q, r0):
        self.q = q
        self.r0 = r0
        self.R = 0.25

    def form(self):
        if self.q == 0:
            q_color = 'black'
        elif self.q < 0:
            q_color = 'blue'
        else:
            q_color = 'red'

        return plt.Circle(self.r0, self.R, color=q_color)

    def compute_field(self, x, y, z):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        Ex = self.const * self.q * (x - self.r0[0]) / r ** 3
        Ey = self.const * self.q * (y - self.r0[1]) / r ** 3
        Ez = 0

        return [Ex, Ey, Ez]

    def compute_potential(self, x, y, z):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        phi = self.const * self.q * (1 / r)

        return [phi, 0, 0]


class Conductor:
    mu_0 = 4 * np.pi * 10E-7
    const = (mu_0 / (2 * np.pi))

    def __init__(self, I, r0):
        self.I = I
        self.R = 0.5
        self.r0 = r0

    def form(self):
        return plt.Circle(self.r0, self.R, edgecolor='black', facecolor='white')

    def details(self):
        if self.I == 0:
            I_direction = '*'
            I_color = 'black'
            size = self.R * 200
        elif self.I > 0:
            I_direction = 'o'
            I_color = 'black'
            size = self.R * 25
        else:
            I_direction = 'x'
            I_color = 'black'
            size = self.R * 200

        return [self.r0[0], self.r0[1], size, I_color, I_direction]

    def compute_field(self, x, y, z):
        mag = self.const * (self.I / np.hypot(x - self.r0[0], y - self.r0[1]))
        Bx = mag * (np.cos(np.arctan2(x - self.r0[0], y - self.r0[1])))
        By = mag * (-np.sin(np.arctan2(x - self.r0[0], y - self.r0[1])))
        Bz = 0

        return [Bx, By, Bz]

    def compute_potential(self, x, y, z):
        mag = -(1 / 2) * self.const * self.I
        Bx = 0
        By = 0
        Bz = mag * np.log((x - self.r0[0])**2 + (y - self.r0[1])**2)

        return [Bx, By, Bz]
