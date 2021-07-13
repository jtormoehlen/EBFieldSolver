import sys

import numpy as np
from matplotlib import pyplot as plt
import imageio as iio

"++++++++++++++++++++++Constants++++++++++++++++++++++"
_c = 299792458.
_pi = np.pi
_mu0 = 4 * _pi * 1E-7
_epsilon0 = 8.85 * 1E-12
_frequency = 500E6
_T = 1 / _frequency
_omega = 2 * np.pi * _frequency
_wavelength = _c / _frequency
_power = 1
_p_norm = np.sqrt(12 * _pi * _c * _power / (_mu0 * _omega ** 4))
_p_0 = np.array([0.0, 0.0, _p_norm])


def dipole_e_vec(x, y, z, p, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r_cross_p = np.cross(r, p)
    rcrossp_cross_r = np.cross(r_cross_p, r)
    r_dot_p = np.dot(3 * r, p)
    r_dot_rdotp = np.dot(r, r_dot_p) - p

    c1 = (_omega ** 3 / (4 * np.pi * _epsilon0 * _c ** 3))
    c2 = (_omega * r_norm) / _c
    c3 = 1 / c2
    c4 = 1 / c2 ** 3
    c5 = 1j / c2 ** 2
    c6 = 1j * (c2 - (_omega * t))

    E = c1 * ((rcrossp_cross_r * c3) + r_dot_rdotp * (c4 - c5)) * np.exp(c6)
    return E


def dipole_b_vec(x, y, z, p, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r_cross_p = np.cross(r, p)

    c1 = (_omega ** 3 / (4 * np.pi * _c ** 2))
    c2 = (_omega * r_norm) / _c
    c3 = 1 / c2
    c4 = 1j / c2 ** 2
    c5 = 1j * (c2 - (_omega * t))

    B = c1 * r_cross_p * (c3 + c4) * np.exp(c5)
    return B


def dipole_poynting_vec(E, B):
    S = np.cross(E, B)
    return S


if __name__ == "__main__":
    nx = 30
    x_max = 2 * _wavelength
    x = np.linspace(-x_max, x_max, nx)

    ny = 30
    y_max = 2 * _wavelength
    y = np.linspace(-y_max, y_max, ny)

    nz = 30
    z_max = 2 * _wavelength
    z = np.linspace(-z_max, z_max, nz)

    X1, Z = np.meshgrid(x, z)
    X2, Y = np.meshgrid(x, y)

    nt = 20
    t0 = 0
    t1 = _T
    t = np.linspace(t0, t1, nt)

    Ex, Ez = np.zeros((len(x), len(z))), np.zeros((len(x), len(z)))
    Bx, By = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Sx, Sz = np.zeros((len(x), len(z))), np.zeros((len(x), len(z)))

    counter = 0
    for dt in t:
        p = _p_0 * np.exp(-1j * _omega * dt)
        for i in range(len(X1)):
            for j in range(len(Z)):
                E = dipole_e_vec(X1[i][j], 0, Z[i][j], p, dt)
                B = dipole_b_vec(X2[i][j], Y[i][j], 0, p, dt)
                S = dipole_poynting_vec(E, B)
                Ex[i][j] = np.real(E[0])
                Ez[i][j] = np.real(E[2])
                Bx[i][j] = np.real(B[0])
                By[i][j] = np.real(B[1])
                Sx[i][j] = np.real(S[0])
                Sz[i][j] = np.real(S[2])

        plt.rcParams['image.cmap'] = 'bwr'
        B_norm = np.hypot(Bx, By)
        plt.quiver(X2, Y, Bx, By)
        # S_norm = np.hypot(Sx, Sz)
        # plt.quiver(X1, Z, Sx, Sz)
        # E_norm = np.hypot(Ex, Ez)
        # plt.quiver(X1, Z, Ex / E_norm, Ez / E_norm, Ez / E_norm)
        # plt.pcolormesh(X1, Z, E_norm, cmap='hot')
        plt.xlabel(r'$x/\lambda$')
        plt.ylabel(r'$y/\lambda$')
        plt.savefig('img/dipole/dipole' + str(counter) + '.png')
        plt.cla()
        counter = counter + 1

    with iio.get_writer('img/dipole.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/dipole/dipole' + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)

    sys.exit(0)
