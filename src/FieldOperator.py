import numpy as np
from scipy.misc import derivative


def grad(f):
    dy, dx = np.gradient(f)
    return [dx, dy]


def gradient(x, y, z, f):
    dx = derivative(lambda x0: f(x0, y, z), x, 0.01)
    dy = derivative(lambda y0: f(x, y0, z), y, 0.01)
    dz = derivative(lambda z0: f(x, y, z0), z, 0.01)
    return [dx, dy, dz]


def divergence(x, y, z, f):
    dx = derivative(lambda x0: f(x0, y, z), x, 0.01)
    dy = derivative(lambda y0: f(x, y0, z), y, 0.01)
    dz = derivative(lambda z0: f(x, y, z0), z, 0.01)
    return dx + dy + dz


def curl(x, y, z, f):
    dfx_dy = derivative(lambda y0: f(x, y0, z)[0], y, 0.01)
    dfx_dz = derivative(lambda z0: f(x, y, z0)[0], z, 0.01)

    dfy_dx = derivative(lambda x0: f(x0, y, z)[1], x, 0.01)
    dfy_dz = derivative(lambda z0: f(x, y, z0)[1], z, 0.01)

    dfz_dx = derivative(lambda x0: f(x0, y, z)[2], x, 0.01)
    dfz_dy = derivative(lambda y0: f(x, y0, z)[2], y, 0.01)

    dx = dfz_dy - dfy_dz
    dy = dfx_dz - dfz_dx
    dz = dfy_dx - dfx_dy

    return [dx, dy, dz]
