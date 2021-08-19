import numpy as np
from scipy.misc import derivative


#
# (*$\nabla \cdot f = \partial_x f \vec{e}_x + \partial_y f \vec{e}_y + \partial_z f \vec{e}_z$*)
#
def gradf(x, y, z, f):
    dx = derivative(lambda x0: f(x0, y, z)[0], x, 0.01)
    dy = derivative(lambda y0: f(x, y0, z)[0], y, 0.01)
    dz = derivative(lambda z0: f(x, y, z0)[0], z, 0.01)
    return np.array([dx, dy, dz])


def grad(F):
    f_x, f_y, f_z = np.gradient(F)
    return np.array([f_x, f_y, f_z])


#
# (*$\nabla \cdot \vec{f} = \partial_x f_x + \partial_y f_y + \partial_z f_z$*)
#
def divf(x, y, z, f):
    dx, dy, dz = gradf(x, y, z, f)
    return dx + dy + dz


#
# (*$\nabla \times \vec{f} = (\partial_y f_z - \partial_z f_y)\vec{e}_x + (\partial_z f_x - \partial_x f_z)\vec{e}_y + (\partial_x f_y - \partial_y f_x)\vec{e}_z$*)
#
def rot(F_x, F_y, F_z):
    f_xx, f_xy, f_xz = np.gradient(F_x)
    f_yx, f_yy, f_yz = np.gradient(F_y)
    f_zx, f_zy, f_zz = np.gradient(F_z)
    f_x = f_zy - f_yz
    f_y = f_xz - f_zx
    f_z = f_yx - f_xy
    return np.array([f_x, f_y, f_z])


def spherical_to_cartesian(x, y, z, v):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    m1 = [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)]
    m2 = [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)]
    m3 = [np.cos(theta), -np.sin(theta), 0]
    return np.array([np.dot(m1, v), np.dot(m2, v), np.dot(m3, v)])
