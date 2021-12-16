import numpy as np


def field(xyz_max, n_xyz, objects, t=-1, function='', indexing='ij'):
    x, y, z = mesh(xyz_max, n_xyz, indexing=indexing)
    x_field, y_field, z_field = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for object in objects:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    f = getattr(object, function)
                    field = np.real(f(x[i][j][k], y[i][j][k], z[i][j][k], t))
                    x_field[i][j][k] += field[0]
                    y_field[i][j][k] += field[1]
                    z_field[i][j][k] += field[2]
    return [x_field, y_field, z_field]


def mesh(xyz_max, n_xyz, indexing='ij'):
    xxx, yyy, zzz = np.meshgrid(np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                np.linspace(-xyz_max, xyz_max, n_xyz),
                                indexing=indexing)
    return xxx, yyy, zzz


def field_round(f_x, f_y, f_xy_min, objects):
    if objects.rod - round(objects.rod) == 0:
        span = 70
    else:
        span = 20
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            f_xy_norm = np.sqrt(f_x[i][j] ** 2 + f_y[i][j] ** 2)
            if f_xy_norm > f_xy_min * span:
                f_x[i][j] = (f_x[i][j] / f_xy_norm) * f_xy_min * span
                f_y[i][j] = (f_y[i][j] / f_xy_norm) * f_xy_min * span
    return f_x, f_y


def r_unit(xy_max, n_xy):
    x, y, z = mesh(xy_max, n_xy)
    plane = round(n_xy / 2)
    phi = np.arctan2(y[:, :, plane], x[:, :, plane])
    e_r = np.array([np.cos(phi), np.sin(phi)])
    return e_r


def phi_unit(xy_max, n_xy):
    x, y, z = mesh(xy_max, n_xy)
    plane = round(n_xy / 2)
    phi = np.arctan2(y[:, :, plane], x[:, :, plane])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    return e_phi


def grad(F):
    f_x, f_y, f_z = np.gradient(F)
    return np.array([f_x, f_y, f_z])


def rot(F_x, F_y, F_z):
    f_xx, f_xy, f_xz = np.gradient(F_x)
    f_yx, f_yy, f_yz = np.gradient(F_y)
    f_zx, f_zy, f_zz = np.gradient(F_z)
    f_x = f_zy - f_yz
    f_y = f_xz - f_zx
    f_z = f_yx - f_xy
    return np.array([f_x, f_y, f_z])
