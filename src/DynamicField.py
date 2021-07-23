import sys

import numpy as np
import FieldUtil as util
import FieldAnimation as anim

"initialize global constants"
_frequency = 500E6
_power = 1
_R = 0.05

_c = 299792458.
_pi = np.pi
_mu0 = 4 * _pi * 1E-7
_epsilon0 = 8.85 * 1E-12
_T = 1 / _frequency
_omega = 2 * np.pi * _frequency
_wavelength = _c / _frequency
_k = (2 * _pi) / _wavelength
_p_norm = np.sqrt(12 * _pi * _c * _power / (_mu0 * _omega ** 4))
_p_0 = np.array([0.0, 0.0, _p_norm])


def E_field(x, y, z, p, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r = r / r_norm
    r_cross_p = np.cross(r, p)
    rcrossp_cross_r = np.cross(r_cross_p, r)
    r_dot_p = np.dot(r, p)
    r_dot_rdotp = np.dot(3 * r, r_dot_p) - p

    c1 = (_omega ** 3 / (4 * np.pi * _epsilon0 * _c ** 3))
    c2 = (_omega * r_norm) / _c
    c3 = 1 / c2
    c4 = 1 / c2 ** 3
    c5 = 1j / c2 ** 2
    c6 = 1j * (c2 - (_omega * t))

    E = c1 * ((rcrossp_cross_r * c3) + (r_dot_rdotp * (c4 - c5))) * np.exp(c6)
    return E


def H_field(x, y, z, p, t):
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


if __name__ == "__main__":
    n_xy = 50
    xy_max = 2 * _wavelength
    x = np.linspace(-xy_max, xy_max, n_xy)
    y = np.linspace(-xy_max, xy_max, n_xy)
    X, Y = np.meshgrid(x, y)

    phi = np.arctan2(Y, X)
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])

    n_t = 50
    t_max = _T
    t = np.linspace(0, t_max, n_t)

    Ex, Ez = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Hx, Hy = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Sx, Sz = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))

    pos = 0
    for dt in t:
        p = _p_0 * np.exp(-1j * _omega * dt)
        for i in range(len(X)):
            for j in range(len(Y)):
                E = E_field(X[i][j], 0, Y[i][j], p, dt)
                H = H_field(X[i][j], Y[i][j], 0, p, dt)
                S = np.cross(E, H)

                Ex[i][j] = np.real(E[0])
                Ez[i][j] = np.real(E[2])
                Hx[i][j] = np.real(H[0])
                Hy[i][j] = np.real(H[1])
                Sx[i][j] = np.real(S[0])
                Sz[i][j] = np.real(S[2])

        util.trim(Ex, Ez, cap=10.0)
        util.plot_arrows(X, Y, Ex, Ez, cmap='winter', cvalue=Ez / np.hypot(Ex, Ez))
        anim.window(labels=[r'$x/$m', r'$z/$m'])
        anim.render_frame(t=t, loc='D_E', pos=pos)

        util.trim(Hx, Hy, cap=0.05)
        util.plot_arrows(X, Y, Hx, Hy, cmap='cool', cvalue=np.array([Hx, Hy]) * e_phi)
        anim.window(labels=[r'$x/$m', r'$y/$m'])
        anim.render_frame(t=t, loc='D_H', pos=pos)

        util.trim(Sx, Sz, cap=0.5)
        util.plot_arrows(X, Y, Sx, Sz, cmap='hot', cvalue=np.array([Sx, Sz]) * e_r)
        anim.window(labels=[r'$x/$m', r'$z/$m'])
        anim.render_frame(t=t, loc='D_S', pos=pos)

        pos += 1

    anim.render_anim(t, 'D_E')
    anim.render_anim(t, 'D_H')
    anim.render_anim(t, 'D_S')

    sys.exit(0)
