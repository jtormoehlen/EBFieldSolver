import numpy as np
import matplotlib.pyplot as plt
import FieldOperator as fo
from mpl_toolkits import mplot3d


def field(x_max, y_max, z_max=0, field_objects=None, function='', nabla='', t=-1):
    x, y, z = mesh(x_max, y_max, z_max)
    x_field, y_field, z_field = np.zeros((len(x), len(x))), np.zeros((len(y), len(y))), np.zeros((len(z), len(z)))
    for field_object in field_objects:
        for i in range(len(x)):
            for j in range(len(y)):
                if nabla == '':
                    if t >= 0:
                        f = getattr(field_object, function)
                        field = np.real(f(x[i][j], y[i][j], z[i][j], t))
                    else:
                        f = getattr(field_object, function)
                        field = f(x[i][j], y[i][j], z[i][j])
                else:
                    operator = getattr(fo, nabla)
                    f = getattr(field_object, function)
                    field = operator(x[i][j], y[i][j], z[i][j], f)
                x_field[i][j] += field[0]
                y_field[i][j] += field[1]
                z_field[i][j] += field[2]
    return [x_field, y_field, z_field]


def mesh(x_max, y_max, z_max=0, n_xyz=50):
    if z_max == 0:
        xx, yy = np.meshgrid(np.linspace(-x_max, x_max, n_xyz),
                             np.linspace(-y_max, y_max, n_xyz))
        zz = np.zeros((len(xx), len(xx)))
    else:
        xx, zz = np.meshgrid(np.linspace(-x_max, x_max, n_xyz),
                             np.linspace(-z_max, z_max, n_xyz))
        yy = np.zeros((len(xx), len(xx)))
    return xx, yy, zz


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


def arrow_field(x_max, y_max, f_x, f_y, cgrad, cmap):
    x, y, z = mesh(x_max, y_max)
    plt.quiver(x, y, f_x, f_y, cgrad, cmap=cmap)


def potential_lines(x_max, y_max, f_xy):
    x, y, z = mesh(x_max, y_max)
    f_xy_levels = np.linspace(np.min(f_xy) / 10, np.max(f_xy) / 10, 4)
    plt.contour(x, y, f_xy, f_xy_levels, colors='k', alpha=0.5)


def field_lines(x_max, y_max, f_x, f_y):
    x, y, z = mesh(x_max, y_max)
    f_xy_norm = np.hypot(f_x, f_y)
    plt.streamplot(x, y, f_x, f_y, color=np.log(f_xy_norm), cmap='cool', zorder=0, density=2)


def forms(field_objects):
    for field_object in field_objects:
        field_object.form()


def radius_unit_vector(x_max, y_max):
    x, y, z = mesh(x_max, y_max)
    phi = np.arctan2(y, x)
    e_r = np.array([np.cos(phi), np.sin(phi)])

    return e_r


def phi_unit_vector(x_max, y_max):
    x, y, z = mesh(x_max, y_max)
    phi = np.arctan2(y, x)
    e_phi = np.array([-np.sin(phi), np.cos(phi)])

    return e_phi


# def contours3d(x, y, f_xy):
#     ax = plt.axes(projection='3d')
#     ax.contour3D(x, y, f_xy, 50, cmap='binary')
