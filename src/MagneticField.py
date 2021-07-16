import sys

import matplotlib.pyplot as plt
import numpy as np
import FieldUtil as utils
import FieldAnimation as anim


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

    def compute_magnetic_field(self, x, y):
        mag = self.const * (self.I / np.hypot(x - self.r0[0], y - self.r0[1]))
        Bx = mag * (-np.sin(np.arctan2(x - self.r0[0], y - self.r0[1])))
        By = mag * (np.cos(np.arctan2(x - self.r0[0], y - self.r0[1])))
        return By, Bx


conductors = []
# single wire
# conductors.append(Conductor(50E-3, [0.0, 0.0]))

# conductor loop
conductors.append(Conductor(-1.0, [0.0, 3.0]))
conductors.append(Conductor(1.0, [0.0, -3.0]))

# coil
# for i in np.linspace(-5, 5, 10):
#     conductors.append(Conductor(-1.0, [i, 3.0]))
#     conductors.append(Conductor(1.0, [i, -3.0]))


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


def compute_details(conductors):
    conductor_details = []
    for conductor in conductors:
        conductor_details.append(conductor.details())
    return conductor_details


if __name__ == "__main__":
    n_xy = 100
    xy_max = 10
    X, Y = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                       np.linspace(-xy_max, xy_max, n_xy))

    Bx, By = compute_total_field(X, Y, conductors)
    total_bodies = compute_bodies(conductors)
    total_details = compute_details(conductors)

    utils.plot_streamlines(X, Y, Bx, By, color=np.log(np.hypot(Bx, By)), cmap='cool', zorder=1, density=2)
    utils.plot_forms(total_bodies)
    utils.plot_details(total_details)

    anim.render_frame(loc='wireloop', aspect=True)

    sys.exit(0)
