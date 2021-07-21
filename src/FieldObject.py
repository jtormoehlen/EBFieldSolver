import numpy as np
from matplotlib import pyplot as plt


class PointCharge:
    epsilon_0 = 8.85E-12
    const = 1 / (4 * np.pi * epsilon_0)

    def __init__(self, q, r0):
        self.q = q
        self.r0 = r0
        self.R = 0.25

    def body(self):
        if self.q == 0:
            q_color = 'black'
        elif self.q < 0:
            q_color = 'blue'
        else:
            q_color = 'red'

        return plt.Circle(self.r0, self.R, color=q_color)

    def compute_field(self, x, y):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        return self.const * self.q * (x - self.r0[0]) / r ** 3, self.const * self.q * (y - self.r0[1]) / r ** 3

    def compute_potential(self, x, y):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        return self.const * self.q * (1 / r)


class Conductor:
    mu_0 = 4 * np.pi * 10E-7
    const = (mu_0 / (2 * np.pi))

    def __init__(self, I, r0):
        self.I = I
        self.R = 0.5
        self.r0 = r0

    def body(self):
        return plt.Circle(self.r0, self.R, edgecolor='black', facecolor='white')

    def details(self):
        if self.I == 0:
            I_direction = 'o'
            I_color = 'black'
            size = self.R * 0
        elif self.I < 0:
            I_direction = 'o'
            I_color = 'black'
            size = self.R * 25
        else:
            I_direction = 'x'
            I_color = 'black'
            size = self.R * 200

        return [self.r0[0], self.r0[1], size, I_color, I_direction]

    def compute_field(self, x, y):
        mag = self.const * (self.I / np.hypot(x - self.r0[0], y - self.r0[1]))
        Bx = mag * (-np.sin(np.arctan2(x - self.r0[0], y - self.r0[1])))
        By = mag * (np.cos(np.arctan2(x - self.r0[0], y - self.r0[1])))
        return By, Bx
