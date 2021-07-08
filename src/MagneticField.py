import sys

import matplotlib.pyplot as plt
import numpy as np


class Conductor:
    mu_0 = 4 * np.pi * 10E-7
    const = (mu_0 / (2 * np.pi))

    def __init__(self, I, r0):
        self.I = I
        self.R = 0.5
        self.r0 = r0

    def body(self):
        if self.I == 0:
            I_color = 'black'
        elif self.I < 0:
            I_color = 'blue'
        else:
            I_color = 'red'

        return plt.Circle(self.r0, self.R, color=I_color)

    def compute_magnetic_field(self, x, y):
        mag = self.const * (self.I / np.hypot(x - self.r0[0], y - self.r0[1]))
        Bx = mag * (-np.sin(np.arctan2(x - self.r0[0], y - self.r0[1])))
        By = mag * (np.cos(np.arctan2(x - self.r0[0], y - self.r0[1])))
        return By, Bx


def compute_total_field(x, y, conductors):
    fields = []
    for conductor in conductors:
        field = conductor.compute_magnetic_field(x, y)
        fields.append(field)

    total_field = np.zeros_like(fields[0])
    for field in fields:
        total_field += field
    return total_field


def compute_bodies(conductors):
    conductor_bodies = []
    for conductor in conductors:
        conductor_bodies.append(conductor.body())
    return conductor_bodies


conductors = []
# single wire
# conductors.append(Conductor(1.0, [0.0, 0.0]))

# conductor loop
# conductors.append(Conductor(1.0, [0.0, 2.0]))
# conductors.append(Conductor(-1.0, [0.0, -2.0]))

# coil
for i in np.linspace(-5, 5, 10):
    conductors.append(Conductor(-1.0, [i, 5.0]))
    conductors.append(Conductor(1.0, [i, -5.0]))

nx, ny = 100, 100
x, y = np.meshgrid(np.linspace(-10, 10, nx), np.linspace(-10, 10, ny))

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$x$')
ax.set_aspect('equal')

Bx, By = compute_total_field(x, y, conductors)
# Bmax = np.hypot(Bx, By)
# ax.quiver(x, y, z, Bx, By, Bz, color='b', length=1, normalize=True)
ax.streamplot(x, y, Bx, By, zorder=1, color=-np.hypot(x, y), cmap='binary')
total_bodies = compute_bodies(conductors)
for body in total_bodies:
    ax.add_patch(body)

plt.savefig('b_field.png')
plt.show()
plt.close(fig)
sys.exit(0)
