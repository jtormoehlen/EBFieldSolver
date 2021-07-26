import sys
import numpy as np
import matplotlib.pyplot as plt
import FieldUtil as util
import FieldAnimation as anim
from FieldObject import HertzDipole

r0 = np.array([0., 0., 0.])
frequency = 500.E6
power = 1.
oscillating_dipoles = []
oscillating_dipoles.append(HertzDipole(r0, frequency, power))

if __name__ == "__main__":
    n_xy = 50
    xy_max = 2 * oscillating_dipoles[0].wavelength
    x = np.linspace(-xy_max, xy_max, n_xy)
    y = np.linspace(-xy_max, xy_max, n_xy)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(x)))

    phi = np.arctan2(Y, X)
    e_r = np.array([np.cos(phi), np.sin(phi)])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    e_z = np.array([0., 0., 1.])

    n_t = 50
    t_max = oscillating_dipoles[0].T
    t = np.linspace(0, t_max, n_t)

    pos = 0
    for dt in t:
        p = oscillating_dipoles[0].p_z * np.exp(-1j * oscillating_dipoles[0].omega * dt) * e_z
        Ex, Ey, Ez = util.total_field(X, Z, Y, oscillating_dipoles, f='E_field', dynamic=True, p=p, dt=dt)
        Hx, Hy, Hz = util.total_field(X, Y, Z, oscillating_dipoles, f='H_field', dynamic=True, p=p, dt=dt)
        Sx, Sy, Sz = util.total_field(X, Z, Y, oscillating_dipoles, f='S_field', dynamic=True, p=p, dt=dt)

        plt.gca().set_facecolor('black')

        util.trim(Ex, Ez, cap=5.0)
        plt.quiver(X, Y, Ex, Ez, Ez / np.hypot(Ex, Ez), cmap='winter')
        anim.window(labels=[r'$x/$m', r'$z/$m'])
        anim.render_frame(t=t, loc='D_E', pos=pos)

        util.trim(Hx, Hy, cap=0.01)
        plt.quiver(X, Y, Hx, Hy, np.array([Hx, Hy]) * e_phi, cmap='cool')
        anim.window(labels=[r'$x/$m', r'$y/$m'])
        anim.render_frame(t=t, loc='D_H', pos=pos)

        util.trim(Sx, Sz, cap=0.05)
        plt.quiver(X, Y, Sx, Sz, np.array([Sx, Sz]) * e_r, cmap='hot')
        anim.window(labels=[r'$x/$m', r'$z/$m'])
        anim.render_frame(t=t, loc='D_S', pos=pos)

        pos += 1

    anim.render_anim(t=t, loc='D_E')
    anim.render_anim(t=t, loc='D_H')
    anim.render_anim(t=t, loc='D_S')

    sys.exit(0)
