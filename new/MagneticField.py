from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

x, y, z = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))

fig = plt.figure()
ax = fig.gca(projection='3d')


def B(x,y):
    I = 1
    mu = 1.26E-6
    mag = (mu / (2 * np.pi)) * (I / np.sqrt(x**2 + y**2))
    Bx = mag * (-np.sin(np.arctan2(y, x)))
    By = mag * (np.cos(np.arctan2(y, x)))
    Bz = z*0
    return Bx, By, Bz


def cylinder(r):
    phi = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


Bx, By, Bz = B(x, y)
cx, cy = cylinder(0.2)

ax.quiver(x, y, z, Bx, By, Bz, color='b', length=1, normalize=True)

for i in np.linspace(-5, 5, 500):
    ax.plot(cx, cy, i, label='Cylinder', color='r')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
