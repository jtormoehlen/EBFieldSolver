import numpy as np


def field(xyz, n_xyz, fobs, t=0, ffunc='', index='ij'):
    """
    Superpose fields (f_x(t), f_y(t), f_z(t)) of field objects.
    :param xyz: list of spatial coords [x1, x2, y1, y2, z1, z2]
    :param n_xyz: grid points
    :param fobs: list of field objects
    :param t: time coord >= 0
    :param ffunc: field function
    :param index: indexing order
    :return: field (f_x, f_y, f_z)
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


def mesh(xyz, n_xyz, index='ij'):
    """
    Mesh object of (x, y, z).
    :param xyz: list of spatial coords [x1, x2, y1, y2, z1, z2]
    :param n_xyz: grid points
    :param index: indexing order
    :return: mesh
    """
    xxx, yyy, zzz = np.meshgrid(np.linspace(xyz[0], xyz[1], n_xyz),
                                np.linspace(xyz[2], xyz[3], n_xyz),
                                np.linspace(xyz[4], xyz[5], n_xyz),
                                indexing=index)
    return xxx, yyy, zzz


def field_round(f, fmin, fobs):
    """
    Limit arrow length in field.
    :param f: field (f_x, f_y)
    :param fmin: minimal length in f
    :param fobs: list of field objects
    :return: limited field
    """
    f_x, f_y = f
    if fobs.rod == 0:
        span = 30
    elif fobs.rod - round(fobs.rod) == 0:
        span = 80
    else:
        span = 50
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            f_xy_norm = np.sqrt(f_x[i][j] ** 2 + f_y[i][j] ** 2)
            if f_xy_norm > fmin * span:
                f_x[i][j] = (f_x[i][j] / f_xy_norm) * fmin * span
                f_y[i][j] = (f_y[i][j] / f_xy_norm) * fmin * span
    return f_x, f_y


def phi_unit(xy, n_xy):
    """
    Azimuthal unit vector.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param n_xy: grid points
    :return: vector phi/|phi|
    """
    x, y, z = mesh(xy, n_xy)
    plane = round(n_xy / 2)
    phi = np.arctan2(y[:, :, plane], x[:, :, plane])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    return e_phi


def grad(F):
    """
    Gradient field.
    :param F: scalar field
    :return: gradient field
    """
    f_x, f_y, f_z = np.gradient(F)
    return np.array([f_x, f_y, f_z])


def rot(F_x, F_y, F_z):
    """
    Vortex field.
    :param F_x: x-comp of field
    :param F_y: y-comp of field
    :param F_z: z-comp of field
    :return: vortex field
    """
    f_xx, f_xy, f_xz = np.gradient(F_x)
    f_yx, f_yy, f_yz = np.gradient(F_y)
    f_zx, f_zy, f_zz = np.gradient(F_z)
    f_x = f_zy - f_yz
    f_y = f_xz - f_zx
    f_z = f_yx - f_xy
    return np.array([f_x, f_y, f_z])
