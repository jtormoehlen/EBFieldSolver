import numpy as np
import matplotlib.pyplot as plt
import FieldOperator as fo
import FieldAnimation as anim
from mpl_toolkits import mplot3d
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d


def field(xy_max, n_xy, plane='xy', objects=None, nabla='', function='', t=-1):
    x, y = mesh(xy_max, n_xy)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(x)
    for object in objects:
        for i in range(len(x)):
            for j in range(len(y)):
                field = field_point(x, y, i, j, plane, object, nabla, function, t)
                x_field[i][j] += field[0]
                y_field[i][j] += field[1]
                z_field[i][j] += field[2]
    return [x_field, y_field, z_field]


def field_point(x, y, i, j, plane, object, nabla, function, t):
    if nabla == '':
        if t >= 0:
            f = getattr(object, function)
            if plane == 'xz':
                field = np.real(f(x[i][j], 0., y[i][j], t))
            else:
                field = np.real(f(x[i][j], y[i][j], 0., t))
        else:
            f = getattr(object, function)
            if plane == 'xz':
                field = f(x[i][j], 0., y[i][j])
            else:
                field = f(x[i][j], y[i][j], 0.)
    else:
        operator = getattr(fo, nabla)
        f = getattr(object, function)
        if plane == 'xz':
            field = operator(x[i][j], 0., y[i][j], f)
        else:
            field = operator(x[i][j], y[i][j], 0., f)
    return field


def mesh(xy_max, n_xy):
    xx, yy = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                         np.linspace(-xy_max, xy_max, n_xy))
    return xx, yy


def field3d(xyz_max, n_xyz, objects=None, nabla='', function='', t=-1):
    x, y, z = mesh3d(xyz_max, n_xyz)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for object in objects:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    field = field_point3d(x, y, z, i, j, k, object, nabla, function, t)
                    x_field[i][j][k] += field[0]
                    y_field[i][j][k] += field[1]
                    z_field[i][j][k] += field[2]
    return [x_field, y_field, z_field]


def field_point3d(x, y, z, i, j, k, object, nabla, function, t):
    if nabla == '':
        if t >= 0:
            f = getattr(object, function)
            field = np.real(f(x[i][j][k], y[i][j][k], z[i][j][k], t))
        else:
            f = getattr(object, function)
            field = f(x[i][j][k], y[i][j][k], z[i][j][k])
    else:
        operator = getattr(fo, nabla)
        f = getattr(object, function)
        field = operator(x[i][j][k], y[i][j][k], z[i][j][k], f)
    return field


def mesh3d(xyz_max, n_xyz):
    xxx, yyy, zzz = np.meshgrid(np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                indexing='ij')
    return xxx, yyy, zzz


def length(f_x, f_y, cap):
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            while f_x[i][j] > cap or f_y[i][j] > cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
            while f_x[i][j] < -cap or f_y[i][j] < -cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
    return f_x, f_y


def arrow_field(xy_max, n_xy, f_x, f_y, normalize=False, cfunc=None, cmap=None):
    x, y = mesh(xy_max, n_xy)
    if normalize:
        f_xy_norm = np.sqrt(f_x ** 2 + f_y ** 2)
        colorf = f_xy_norm
    else:
        f_xy_norm = 1.
        colorf = np.sqrt(f_x ** 2 + f_y ** 2)

    anim.aspect_ratio()
    if cfunc is None:
        plt.quiver(x, y,
                   f_x / f_xy_norm, f_y / f_xy_norm,
                   np.log(colorf), cmap='cool')
    else:
        anim.background_color('black')
        plt.quiver(x, y,
                   f_x / f_xy_norm, f_y / f_xy_norm,
                   cfunc, cmap=cmap)


def arrow_field3d(xyz_max, n_xyz, f_x, f_y, f_z):
    x, y, z = mesh3d(xyz_max, n_xyz)
    anim.aspect_ratio(False)
    plt.subplot(projection='3d', label='default')
    plt.quiver(x, y, z, f_x, f_y, f_z,
               length=xyz_max / n_xyz, normalize=True)
    ring = Circle((0, 0), 1, edgecolor='grey', fill=False)
    plt.gca().add_patch(ring)
    art3d.pathpatch_2d_to_3d(ring, z=0, zdir="x")


def potential_lines(xy_max, n_xy, f_xy):
    x, y = mesh(xy_max, n_xy)
    f_xy_levels = np.linspace(np.min(f_xy) / 10, np.max(f_xy) / 10, 4)
    anim.aspect_ratio()
    plt.contour(x, y, f_xy, f_xy_levels, colors='k', alpha=0.5)


def field_lines(xy_max, n_xy, f_x, f_y):
    x, y = mesh(xy_max, n_xy)
    f_xy_norm = np.hypot(f_x, f_y)
    anim.aspect_ratio()
    plt.streamplot(x, y, f_x, f_y, color=np.log(f_xy_norm), cmap='cool', zorder=0, density=2)


def forms(field_objects):
    for field_object in field_objects:
        field_object.form()


def radius_unit_vector(xy_max, n_xy):
    x, y = mesh(xy_max, n_xy)
    phi = np.arctan2(y, x)
    e_r = np.array([np.cos(phi), np.sin(phi)])

    return e_r


def phi_unit_vector(xy_max, n_xy):
    x, y = mesh(xy_max, n_xy)
    phi = np.arctan2(y, x)
    e_phi = np.array([-np.sin(phi), np.cos(phi)])

    return e_phi
