import numpy as np
from scipy.misc import derivative


def gradient(f):
    dy, dx = np.gradient(f)
    return [dx, dy]


# def divergence(f):
#     num_dims = len(f)
#     return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def divergence(x, y, f):
    dx = derivative(lambda x0: f(x0, y), x, 0.01)
    dy = derivative(lambda y0: f(x, y0), y, 0.01)
    return dx + dy


def curl_x(x, y, field_object):
    dx = derivative(lambda y0: field_object.compute_potential(x, y0), y, 0.01)
    dy = derivative(lambda x0: -field_object.compute_potential(x0, y), x, 0.01)
    return dx, dy
