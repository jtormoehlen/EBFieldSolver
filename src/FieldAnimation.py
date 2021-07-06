import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from ElectricField import init_plot, get_charges, compute_total_field, compute_total_potentials


def init():
    total_potentials = compute_total_potentials(x, y, charges)
    pots = ax.contourf(x, y, total_potentials, z, cmap=plt.get_cmap('coolwarm'))
    return []


def update(i, pots, x, y):
    ax.collections = []
    # ax.patches = []
    charges[0].r0 = [random.randrange(-2, 2), random.randrange(-2, 2)]
    # charge_artists(ax)
    total_potentials = compute_total_potentials(x, y, charges)
    pots = ax.contourf(x, y, total_potentials, z, cmap=plt.get_cmap('coolwarm'))
    return []


nx, ny = 64, 64
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set(xlim=(-5, 5), ylim=(-5, 5))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_aspect('equal')

charges = get_charges()
total_potentials = compute_total_potentials(x, y, charges)
z = np.linspace(np.min(total_potentials) / 2, np.max(total_potentials) / 2, 20)
pots = ax.contourf(x, y, total_potentials, z, cmap=plt.get_cmap('coolwarm'))

anim = animation.FuncAnimation(fig, update, fargs=(pots, x, y), init_func=init, interval=100, blit=True)

plt.savefig('field.png')
plt.show()
