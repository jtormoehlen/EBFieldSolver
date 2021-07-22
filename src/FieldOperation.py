import numpy as np


def gradient(f):
    dy, dx = np.gradient(f)
    return [dx, dy]


def divergence(f):
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def curl(x, y, z, u, v, w):
    dx = x[0, :, 0]
    dy = y[:, 0, 0]
    dz = z[0, 0, :]

    dummy, dFx_dy, dFx_dz = np.gradient(u, dx, dy, dz, axis=[1, 0, 2])
    dFy_dx, dummy, dFy_dz = np.gradient(v, dx, dy, dz, axis=[1, 0, 2])
    dFz_dx, dFz_dy, dummy = np.gradient(w, dx, dy, dz, axis=[1, 0, 2])

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    l = np.sqrt(np.power(u, 2.0) + np.power(v, 2.0) + np.power(w, 2.0))

    m1 = np.multiply(rot_x, u)
    m2 = np.multiply(rot_y, v)
    m3 = np.multiply(rot_z, w)

    tmp1 = (m1 + m2 + m3)
    tmp2 = np.multiply(l, 2.0)

    av = np.divide(tmp1, tmp2)

    return rot_x, rot_y, rot_z, av
