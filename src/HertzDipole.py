import sys

import numpy as np
import FieldUtil as util
import matplotlib.colors as colors

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

    nt = 25
    t0 = 0
    t1 = _T
    t = np.linspace(t0, t1, nt)

    Ex, Ez = np.zeros((len(x), len(z))), np.zeros((len(x), len(z)))
    Hx, Hy = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    Sx, Sz = np.zeros((len(x), len(z))), np.zeros((len(x), len(z)))

    x_0 = np.linspace(1E-3, 8 * _wavelength, 100)

    E_xz_0, H_xy_0 = np.zeros(len(x_0)), np.zeros((len(x_0)))

    counter = 0
    for dt in t:
        p = _p_0 * np.exp(-1 * 1j * _omega * dt)
        for i in range(len(X1)):
            for j in range(len(Z)):

                # if np.hypot(X1[i][j], Z[i][j]) <= _R:
                #     E = np.array([0.0, 0.0, 0.0])
                #     H = np.array([0.0, 0.0, 0.0])
                #     S = np.array([0.0, 0.0, 0.0])
                # else:
                #     E = dipole_E(X1[i][j], 0, Z[i][j], p, dt)
                #     H = dipole_H(X2[i][j], Y[i][j], 0, p, dt)
                #     S = dipole_Poynting(E, H)

                E = dipole_E(X1[i][j], 0, Z[i][j], p, dt)
                H = dipole_H(X2[i][j], Y[i][j], 0, p, dt)
                S = dipole_Poynting(E, H)

                Ex[i][j] = np.real(E[0])
                Ez[i][j] = np.real(E[2])
                Hx[i][j] = np.real(H[0])
                Hy[i][j] = np.real(H[1])
                Sx[i][j] = np.real(S[0])
                Sz[i][j] = np.real(S[2])

        for k in range(len(x_0)):
            E = dipole_E(x_0[k], 0, 0, p, dt)
            H = dipole_H(x_0[k], 0, 0, p, dt)
            E_xz_0[k] = np.real(dipole_E(x_0[k], 0, 0, p, dt)[2])
            H_xy_0[k] = np.real(dipole_H(x_0[k], 0, 0, p, dt)[1])

        # S_norm = np.hypot(Sx, Sz)
        # plt.quiver(X1, Z, Sx / S_norm, Sz / S_norm)
        # plt.contour(X1, Z, E_norm, levels=np.linspace(2, 10, 4))
        # plt.pcolormesh(X1, Z, E_norm, cmap='hot')

        util.plot_arrows(X1, Z, Ex, Ez, cmap='winter', cap=5.0)
        # util.plot_intensity(X1, Z, np.hypot(Ex, Ez))
        # util.plot_contour(X1, Z, np.hypot(Ex, Ez))
        util.render_frame(r'$x/\lambda$', r'$z/\lambda$', counter, t, 'dipole_E')

        util.plot_arrows(X2, Y, Hx, Hy, cmap='cool', cap=0.05)
        util.render_frame(r'$x/\lambda$', r'$y/\lambda$', counter, t, 'dipole_H')

        # S_dz, S_dx = np.gradient(np.hypot(Sx, Sz))
        util.plot_arrows(X1, Z, Sx, Sz, cmap='hot', cap=0.1)
        util.render_frame(r'$x/\lambda$', r'$z/\lambda$', counter, t, 'dipole_S')

        util.plot_normal(x_0, -E_xz_0)
        util.plot_normal(x_0, 1000*H_xy_0)
        util.render_frame(r'$x/$m', r'$E_{z=0}$ and $H_{y=0}$', counter, t, 'E_xz_0_H_xy_0', [0, 6], [-100, 100], aspect=False)

        counter = counter + 1

    util.render_anim(t, 'dipole_H')
    util.render_anim(t, 'dipole_E')
    util.render_anim(t, 'dipole_S')
    util.render_anim(t, 'E_xz_0_H_xy_0')

    sys.exit(0)
