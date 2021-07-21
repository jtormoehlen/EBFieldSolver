import sys

import numpy as np
import FieldUtil as util
import FieldAnimation as anim

"CONSTANTS"
_frequency = 500E6
_power = 1

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

_R = 0.05
_q = 1E-10
_d0 = _wavelength / 2


def dipole_E(x, y, z, p, t):
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


def antenna_E(x, y, z, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r = r / r_norm

    t_r = t - (r_norm / _c)

    p = np.array([0.0, 0.0, _q * _d0 * np.sin(_omega * t_r)])
    p_dot = np.array([0.0, 0.0, _omega * _q * _d0 * np.cos(_omega * t_r)])
    p_dotdot = np.array([0.0, 0.0, _omega**2 * _q * _d0 * -np.sin(_omega * t_r)])

    c1 = 1 / (4 * _pi * _epsilon0)
    c2 = (np.dot(3 * r, np.dot(p, r)) / r_norm ** 3) - (p / r_norm ** 3)
    c3 = (np.dot(3 * r, np.dot(p_dot, r)) / (_c * r_norm ** 2)) - (p_dot / (_c * r_norm ** 2))
    c4 = (np.dot(r, np.dot(p_dotdot, r)) / (_c ** 2 * r_norm)) - (p_dotdot / (_c ** 2 * r_norm))

    E = c1 * (c2 + c3 + c4)

    return E


def plot_E(x, y, t):
    const = (_omega * t) - ((_omega * np.hypot(x, y)) / _c)
    ex1 = - ((x * np.sin(const)) / (4 * (x**2 + y**2) * _pi))
    ex2 = ((x * np.cos(const)) / (4 * (x**2 + y**2)**(3/2) * _pi))
    return ex1 + ex2


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


def antenna_B(x, y, z, t):
    r = np.array([x, y, z])
    r_norm = np.linalg.norm(r)
    r = r / r_norm

    t_r = t - (r_norm / _c)

    p_dot = np.array([0.0, 0.0, _omega * _q * _d0 * np.cos(_omega * t_r)])
    p_dotdot = np.array([0.0, 0.0, _omega ** 2 * _q * _d0 * -np.sin(_omega * t_r)])

    c1 = 1 / (4 * _pi * _epsilon0 * _c ** 2 * r_norm ** 2)
    c2 = np.cross(p_dot, r)
    c3 = (r_norm / _c) * np.cross(p_dotdot, r)

    B = c1 * (c2 + c3)

    return B


def dipole_Poynting(E, H):
    # Sx = E[2] * H[1]
    # Sz = E[0] * H[0]
    # return np.array([Sx, 0.0, Sz])
    return np.cross(E, H)


if __name__ == "__main__":
    n_xy = 50
    xy_max = 2 * _wavelength
    x = np.linspace(-xy_max, xy_max, n_xy)
    y = np.linspace(-xy_max, xy_max, n_xy)
    X, Y = np.meshgrid(x, y, indexing='ij')

    phi = np.arctan2(Y, X)
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])

    n_t = 50
    t_max = _T
    t = np.linspace(0, t_max, n_t)

    Ex, Ez = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Hx, Hy = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Sx, Sz = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    E_S = np.zeros((len(x), len(y)))

    x = np.linspace(_R, 10 * _wavelength, 200)
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

        # Ez_mean = np.mean(np.sqrt(Ez ** 2))
        # util.plot_contour(X, Y, np.sqrt(Ez ** 2), levels=np.linspace(0, 2 * Ez_mean, 5))
        # util.plot_intensity(X, Y, np.hypot(Ex, Ez))
        # util.plot_contour3d(X, Y, np.hypot(Ex, Ez))
        util.trim(Ex, Ez, cap=10.0)
        util.plot_arrows(X, Y, Ex, Ez, cmap='winter', cvalue=Ez / np.hypot(Ex, Ez))
        anim.render_frame(r'$x/$m', r'$z/$m', counter, t, 'D_E')

        util.trim(Hx, Hy, cap=0.05)
        util.plot_arrows(X, Y, Hx, Hy, cmap='cool', cvalue=np.array([Hx, Hy]) * e_phi)
        anim.render_frame(r'$x/$m', r'$y/$m', counter, t, 'D_H')

        util.trim(Sx, Sz, cap=0.5)
        util.plot_arrows(X, Y, Sx, Sz, cmap='hot', cvalue=np.array([Sx, Sz]) * e_r)
        anim.render_frame(r'$x/$m', r'$z/$m', counter, t, 'D_S')

        util.plot_normal(x, -Ez0)
        util.plot_normal(x, 1000 * Hy0)
        # util.plot_normal(x, 100*Sz0)
        anim.render_frame(r'$x/$m', r'$E_{z=0}$ and $10^3 \cdot H_{y=0}$', counter, t, 'Ez0_Hy0', [0, 5.0], [-100, 100], aspect=False)

        counter = counter + 1

    anim.render_anim(t, 'D_E')
    anim.render_anim(t, 'D_H')
    anim.render_anim(t, 'D_S')
    anim.render_anim(t, 'Ez0_Hy0')

    sys.exit(0)
