import numpy as np
import matplotlib.pyplot as plt
import FieldOperator as fo
from mpl_toolkits import mplot3d


def total_field(x, y, z, field_objects, f='', dynamic=False, p=None, dt=0):
    x_field, y_field, z_field = np.zeros((len(x), len(x))), np.zeros((len(y), len(y))), np.zeros((len(z), len(z)))
    for field_object in field_objects:
        for i in range(len(x)):
            for j in range(len(y)):
                if dynamic:
                    function = getattr(field_object, f)
                    field = np.real(function(x[i][j], y[i][j], z[i][j], p, dt))
                else:
                    function = getattr(field_object, f)
                    field = function(x[i][j], y[i][j], z[i][j])
                x_field[i][j] += field[0]
                y_field[i][j] += field[1]
                z_field[i][j] += field[2]
    return [x_field, y_field, z_field]


def total_diff(x, y, z, field_objects, f='', nabla=''):
    nabla_fx, nabla_fy, nabla_fz = np.zeros((len(x), len(x))), np.zeros((len(y), len(y))), np.zeros((len(z), len(z)))
    for i in range(len(x)):
        for j in range(len(y)):
            for field_object in field_objects:
                operator = getattr(fo, nabla)
                function = getattr(field_object, f)
                nabla_x, nabla_y, nabla_z = operator(x[i][j], y[i][j], z[i][j], function)
                nabla_fx[i][j] += nabla_x
                nabla_fy[i][j] += nabla_y
                nabla_fz[i][j] += nabla_z
    return [nabla_fx, nabla_fy, nabla_fz]


def arrows(x, y, f_x, f_y, normalize=False, colorgradient=0):
    if normalize:
        f_norm = np.hypot(f_x, f_y)
    else:
        f_norm = 1.0
    plt.quiver(x, y, f_x / f_norm, f_y / f_norm, colorgradient)


def trim(f_x, f_y, cap):
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            while f_x[i][j] > cap or f_y[i][j] > cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
            while f_x[i][j] < -cap or f_y[i][j] < -cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
    return f_x, f_y


def contours(x, y, f_xy, levels=None):
    plt.contour(x, y, f_xy, levels=levels, colors='k', alpha=0.5)


def contoursf(x, y, f_xy, levels=None, cmap='bwr'):
    plt.contourf(x, y, f_xy, levels, cmap=cmap)


def contours3d(x, y, f_xy):
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, f_xy, 50, cmap='binary')


def streamlines(x, y, f_x, f_y, color='black', cmap=None, zorder=1, density=2):
    plt.streamplot(x, y, f_x, f_y, color=color, cmap=cmap, zorder=zorder, density=density)


def intensities(x, y, f_xy, cmap='hot'):
    plt.pcolormesh(x, y, f_xy, cmap=cmap)


def forms(field_objects):
    for field_object in field_objects:
        plt.gca().add_patch(field_object.form())


def details(field_objects):
    details = []
    for field_object in field_objects:
        details.append(field_object.details())
    for d in details:
        plt.scatter(d[0], d[1], d[2], d[3], d[4], zorder=3)
