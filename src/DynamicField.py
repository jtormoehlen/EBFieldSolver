import sys
import numpy as np
import matplotlib.pyplot as plt
import FieldAnimation as anim
from FieldObject import HertzDipole
from FieldUtil import field, trim, arrow_field, radius_unit_vector, phi_unit_vector, forms

r0 = np.array([0., 0., 0.])
frequency = 500.E6
power = 1.
dipole = []
dipole.append(HertzDipole(r0, frequency, power))

if __name__ == "__main__":
    xy_max = 2 * dipole[0].wavelength

    n_t = 50
    t_max = dipole[0].T
    t = np.linspace(0, t_max, n_t)

    pos = 0
    for dt in t:
        Ex, Ey, Ez = field(xy_max, xy_max, xy_max, field_objects=dipole, function='E_field', t=dt)
        Hx, Hy, Hz = field(xy_max, xy_max, field_objects=dipole, function='H_field', t=dt)
        Sx, Sy, Sz = field(xy_max, xy_max, xy_max, field_objects=dipole, function='S_field', t=dt)

        anim.background('black')

        trim(Ex, Ez, cap=5.0)
        arrow_field(xy_max, xy_max, Ex, Ez, Ez / np.hypot(Ex, Ez), cmap='winter')
        forms(dipole)
        anim.axes(labels=[r'$x/$m', r'$z/$m'])
        anim.save_frame(t=t, loc='D_E', pos=pos)

        e_phi = phi_unit_vector(xy_max, xy_max)
        trim(Hx, Hy, cap=0.01)
        arrow_field(xy_max, xy_max, Hx, Hy, np.array([Hx, Hy]) * e_phi, cmap='cool')
        forms(dipole)
        anim.axes(labels=[r'$x/$m', r'$y/$m'])
        anim.save_frame(t=t, loc='D_H', pos=pos)

        e_r = radius_unit_vector(xy_max, xy_max)
        trim(Sx, Sz, cap=0.05)
        arrow_field(xy_max, xy_max, Sx, Sz, np.array([Sx, Sz]) * e_r, cmap='hot')
        forms(dipole)
        anim.axes(labels=[r'$x/$m', r'$z/$m'])
        anim.save_frame(t=t, loc='D_S', pos=pos)

        pos += 1

    anim.save_anim(t=t, loc='D_E')
    anim.save_anim(t=t, loc='D_H')
    anim.save_anim(t=t, loc='D_S')

    sys.exit(0)
