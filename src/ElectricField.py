# Import required modules
import sys

import numpy as np
import matplotlib.pyplot as plt


class PointCharge:
    epsilon_0 = 8.85E-12
    const = 1 / (4 * np.pi * epsilon_0)

    def __init__(self, q, r0):
        self.q = q
        self.r0 = r0
        self.R = 0.1

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

    def compute_potentials(self, x, y):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        return self.const * self.q * (1 / r)


# setup charges with loc and q
charges = []

# dy = np.arange(-4.0, 4.0, 0.05)
# for i in dy:
#     charges.append(PointCharge(10.0, [-1.0, i]))
#
# for i in dy:
#     charges.append(PointCharge(-10.0, [1.0, i]))

# dipole charge
# charges.append(PointCharge(1.0, [0.0, -1.0]))
# charges.append(PointCharge(-1.0, [0.0, 1.0]))

# triple charge
# a = 4.0
# charges.append(PointCharge(-1.0, [0.0, 0.0]))
# charges.append(PointCharge(-1.0, [0.0, 0.0]))
# charges.append(PointCharge(1.0, [a, 0.0]))
# charges.append(PointCharge(-1.0, [0.0, a]))

# quadrupole charge
charges.append(PointCharge(-1.0, [-1.0, 1.0]))
charges.append(PointCharge(1.0, [1.0, 1.0]))
charges.append(PointCharge(1.0, [-1.0, -1.0]))
charges.append(PointCharge(-1.0, [1.0, -1.0]))

# single positive charge
# charges.append(PointCharge(-1.0, [0.0, 0.0]))


def compute_total_field(x, y, charges):
    fields = []
    for charge in charges:
        fields.append(charge.compute_field(x, y))

    total_field = np.zeros_like(fields[0])
    for field in fields:
        total_field += field
    return total_field


def norm_total_field(x, y, charges):
    field_x, field_y = compute_total_field(x, y, charges)
    return np.hypot(field_x, field_y)


def compute_total_potentials(x, y, charges):

    potentials = []
    for charge in charges:
        potentials.append(charge.compute_potentials(x, y))

    total_potentials = np.zeros_like(potentials[0])
    for potential in potentials:
        total_potentials += potential
    return total_potentials


def compute_bodies(charges):
    charge_bodies = []
    for charge in charges:
        # masking
        charge_bodies.append(charge.body())
    return charge_bodies


nx, ny = 100, 100
x, y = np.meshgrid(np.linspace(-5, 5, nx), np.linspace(-5, 5, ny))

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
# ax.set(xlim=(-5, 5), ylim=(-5, 5))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_aspect('equal')

Ex, Ey = compute_total_field(x, y, charges)
# norm_field = norm_total_field(x, y, charges)
# ax.quiver(x, y, field_x / norm_field, field_y / norm_field, color='black', scale=25)
ax.streamplot(x, y, Ex, Ey, color='black')

total_potentials = compute_total_potentials(x, y, charges)
z = np.linspace(np.min(total_potentials) / 10, np.max(total_potentials) / 10, 10)
# ax.contour(x, y, total_potentials, z, colors='k')
ax.contourf(x, y, total_potentials, z, cmap='bwr')
# ax.clabel(pots, inline=True, fontsize=10)

total_bodies = compute_bodies(charges)
for body in total_bodies:
    ax.add_patch(body)

plt.savefig('e_field.png')
plt.show()
plt.close(fig)
sys.exit(0)
