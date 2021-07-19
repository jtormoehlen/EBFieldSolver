import sys

import numpy as np
import FieldUtil as util
import FieldAnimation as anim

"++++++++++++++++++++++Constants++++++++++++++++++++++"
_frequency = 500E6
_power = 1
_R = 0.1

_c = 299792458.
_pi = np.pi
_mu0 = 4 * _pi * 1E-7
_epsilon0 = 8.85 * 1E-12
_T = 1 / _frequency
_omega = 2 * np.pi * _frequency
_wavelength = _c / _frequency
_p_norm = np.sqrt(12 * _pi * _c * _power / (_mu0 * _omega ** 4))
_p_0 = np.array([0.0, 0.0, _p_norm])


def dipole_E(x, y, z, p, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r = r / r_norm
    r_cross_p = np.cross(r, p)
    rcrossp_cross_r = np.cross(r_cross_p, r)
    r_dot_p = np.dot(r, p)
    r_dot_rdotp = (3 * r * r_dot_p) - p

    c1 = (_omega ** 3 / (4 * np.pi * _epsilon0 * _c ** 3))
    c2 = (_omega * r_norm) / _c
    c3 = 1 / c2
    c4 = 1 / c2 ** 3
    c5 = 1j / c2 ** 2
    c6 = 1j * (c2 - (_omega * t))

    E = c1 * ((rcrossp_cross_r * c3) + (r_dot_rdotp * (c4 - c5))) * np.exp(c6)

    return E


def dipole_H(x, y, z, p, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r = r / r_norm
    r_cross_p = np.cross(r, p)

    c1 = (_omega ** 3 / (4 * np.pi * _c ** 2))
    c2 = (_omega * r_norm) / _c
    c3 = 1 / c2
    c4 = 1j / c2 ** 2
    c5 = 1j * (c2 - (_omega * t))

    H = c1 * r_cross_p * (c3 + c4) * np.exp(c5)
    return H


def dipole_Poynting(E, H):
    S = np.cross(E, H)
    return S


if __name__ == "__main__":
    n_xy = 100
    xy_max = 1 * _wavelength
    x = np.linspace(-xy_max, xy_max, n_xy)
    y = np.linspace(-xy_max, xy_max, n_xy)
    X, Y = np.meshgrid(x, y)
    y_min = np.min(abs(y))

    n_t = 25
    t_max = _T
    t = np.linspace(0, t_max, n_t)

    Ex, Ez = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Hx, Hy = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Sx, Sz = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))

    x = np.linspace(1E-3, 8 * _wavelength, 200)
    Ez0, Hy0, Sz0 = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))

    counter = 0
    for dt in t:
        p = _p_0 * np.exp(-1 * 1j * _omega * dt)
        for i in range(len(X)):
            for j in range(len(Y)):
                E = dipole_E(X[i][j], 0, Y[i][j], p, dt)
                H = dipole_H(X[i][j], Y[i][j], 0, p, dt)
                S = dipole_Poynting(E, H)

                Ex[i][j] = np.real(E[0])
                Ez[i][j] = np.real(E[2])
                Hx[i][j] = np.real(H[0])
                Hy[i][j] = np.real(H[1])
                Sx[i][j] = np.real(S[0])
                Sz[i][j] = np.real(S[2])

        for k in range(len(x)):
            E = dipole_E(x[k], 0, 0, p, dt)
            H = dipole_H(x[k], 0, 0, p, dt)
            S = dipole_Poynting(E, H)
            Ez0[k] = np.real(E[2])
            Hy0[k] = np.real(H[1])
            Sz0[k] = np.real(S[0])

        util.plot_arrows(X, Y, Ex, Ez, cmap='winter', cap=5.0)
        # util.plot_contour(X1, Z, E_norm, levels=np.linspace(2, 10, 4))
        anim.render_frame(r'$x/\lambda$', r'$z/\lambda$', counter, t, 'D_E')

        util.plot_arrows(X, Y, Hx, Hy, cmap='cool', cap=0.05)
        anim.render_frame(r'$x/\lambda$', r'$y/\lambda$', counter, t, 'D_H')

        util.plot_arrows(X, Y, Sx, Sz, cmap='hot', cap=0.1)
        anim.render_frame(r'$x/\lambda$', r'$z/\lambda$', counter, t, 'D_S')

        util.plot_normal(x, -Ez0)
        util.plot_normal(x, 1000*Hy0)
        # util.plot_normal(x, 100*Sz0)
        anim.render_frame(r'$x/$m', r'$E_{z=0}$ and $H_{y=0}$', counter, t, 'Ez0_Hy0', [0, 5.0], [-100, 100], aspect=False)

        counter = counter + 1

    anim.render_anim(t, 'D_E')
    anim.render_anim(t, 'D_H')
    anim.render_anim(t, 'D_S')
    anim.render_anim(t, 'Ez0_Hy0')

    sys.exit(0)
