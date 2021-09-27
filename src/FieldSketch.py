import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle


def draw_cylinder(radius, height, elevation, resolution, color, x_center, y_center):
    x = np.linspace(x_center - radius, x_center + radius, resolution)
    z = np.linspace(elevation, elevation + height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius ** 2 - (X - x_center) ** 2) + y_center  # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2 * y_center - Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation + height, zdir="z")


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.quiver(0., 0., 0., 5., 5., 5., color='black')
limits = 10
ax.set_xlim([-limits, limits])
ax.set_ylim([-limits, limits])
ax.set_zlim([-limits, limits])

# params
radius = 0.2
height = 4
elevation = 0.1
resolution = 25
color = 'grey'
x_center = 0
y_center = 0

draw_cylinder(radius, height, elevation, resolution, color, x_center, y_center)
draw_cylinder(radius, -height, -elevation, resolution, color, x_center, y_center)

# ax.grid(False)
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
# ax.axes.zaxis.set_ticklabels([])
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_zlabel(r'$z$')

ax.set_axis_off()

plt.show()
