import numpy as np


def field(xyz, n_xyz, fobs, t=-1, ffunc='', index='ij'):
    """
    Iterator for spatial and temporal field coords.
    :param xyz: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param n_xyz: number of grid points in -xyz_max to xyz_max
    :param fobs: list contains all field emitting objects
    :param t: time coord; default: t=0
    :param ffunc: desired field to compute
    :param index: indexing order for mesh; default: line before column
    :return: real part of desired superposed field at time t
    """
    x, y, z = mesh(xyz, n_xyz, index=index)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for fob in fobs:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    f = getattr(fob, ffunc)
                    field = np.real(f(x[i][j][k], y[i][j][k], z[i][j][k], t))
                    x_field[i][j][k] += field[0]
                    y_field[i][j][k] += field[1]
                    z_field[i][j][k] += field[2]
    return [x_field, y_field, z_field]


def mesh(xyz, n_xyz, index='ij', plot3d=True):
    """
    Creates mesh object from three one-dimensional arrays of spatial coords.
    :param plot3d:
    :param xyz: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param n_xyz: number of grid points in -xyz_max to xyz_max
    :param index: indexing order for mesh; default: line before column
    :return: three 3d-arrays containing spatial coords (mesh)
    """
    if plot3d:
        xxx, yyy, zzz = np.meshgrid(np.linspace(xyz[0], xyz[1], n_xyz),
                                    np.linspace(xyz[2], xyz[3], n_xyz),
                                    np.linspace(xyz[4], xyz[5], n_xyz),
                                    indexing=index)
        return xxx, yyy, zzz
    else:
        xx, yy = np.meshgrid(np.linspace(xyz[0], xyz[1], n_xyz),
                             np.linspace(xyz[2], xyz[3], n_xyz),
                             indexing=index)
        return xx, yy


def field_round(f_x, f_y, f_xy_min, fobs):
    """
    Capping vector length of given field components.
    :param f_x: 3d-grid of x-coord
    :param f_y: 3d-grid of y-coord
    :param f_xy_min: minimum-length of a vector in field
    :param fobs: list contains all field emitting objects
    :return: field with capped vector length
    """

    if fobs.rod == 0:
        span = 30
    elif fobs.rod - round(fobs.rod) == 0:
        span = 80
    else:
        span = 50
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            f_xy_norm = np.sqrt(f_x[i][j] ** 2 + f_y[i][j] ** 2)
            if f_xy_norm > f_xy_min * span:
                f_x[i][j] = (f_x[i][j] / f_xy_norm) * f_xy_min * span
                f_y[i][j] = (f_y[i][j] / f_xy_norm) * f_xy_min * span
    return f_x, f_y


def phi_unit(xy, n_xy):
    """
    Angle Unit vector for 2d grid.
    :param xy: spatial coords (x, y) all from -xy_max to xy_max
    :param n_xy: number of grid points in -xy_max to xy_max
    :return: phi unit vector for all grid points
    """
    x, y, z = mesh(xy, n_xy)
    plane = round(n_xy / 2)
    phi = np.arctan2(y[:, :, plane], x[:, :, plane])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    return e_phi


def grad(F):
    """
    Computes gradient field using numpy's gradient function.
    :param F: scalar field
    :return: gradient field
    """
    f_x, f_y, f_z = np.gradient(F)
    return np.array([f_x, f_y, f_z])


def rot(F_x, F_y, F_z):
    """
    Computes vortex field using numpy's gradient function.
    :param F_x: 3d-grid of x-coord
    :param F_y: 3d-grid of y-coord
    :param F_z: 3d-grid of z-coord
    :return: vortex field
    """
    f_xx, f_xy, f_xz = np.gradient(F_x)
    f_yx, f_yy, f_yz = np.gradient(F_y)
    f_zx, f_zy, f_zz = np.gradient(F_z)
    f_x = f_zy - f_yz
    f_y = f_xz - f_zx
    f_z = f_yx - f_xy
    return np.array([f_x, f_y, f_z])
