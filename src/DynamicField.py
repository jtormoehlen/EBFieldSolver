import sys
import numpy as np
import FieldAnimation as anim
from FieldObject import HertzDipole
from FieldUtil import field, arrow_field, phi_unit_vector, radius_unit_vector, length

r0 = np.array([0., 0., 0.])
frequency = 500.E6
power = 1.
dipole = []
dipole.append(HertzDipole(r0, frequency, power))

if __name__ == "__main__":
    xyz_max = 2 * dipole[0].wavelength
    n_xyz = 30

    n_t = 50
    t_max = dipole[0].T
    t = np.linspace(0, t_max, n_t)

    pos = 0
    for dt in t:
        Ex, Ey, Ez = field(xyz_max, n_xyz, plane='xz', objects=dipole, function='E', t=dt)
        Hx, Hy, Hz = field(xyz_max, n_xyz, plane='xy', objects=dipole, function='H', t=dt)
        Sx, Sy, Sz = field(xyz_max, n_xyz, plane='xz', objects=dipole, function='S', t=dt)

        length(Ex, Ez, cap=2.5)
        arrow_field(xyz_max, n_xyz, Ex, Ez,
                    cfunc=Ez/np.hypot(Ex, Ez), cmap='winter')
        anim.axes(x_label=r'$x/$m', y_label=r'$z/$m')
        anim.save_frame(t=t, loc='E', pos=pos)

        e_phi = phi_unit_vector(xyz_max, n_xy=n_xyz)
        length(Hx, Hy, cap=0.01)
        arrow_field(xyz_max, n_xyz, Hx, Hy,
                    cfunc=np.array([Hx, Hy]) * e_phi, cmap='cool')
        anim.axes(x_label=r'$x/$m', y_label=r'$y/$m')
        anim.save_frame(t=t, loc='H', pos=pos)

        e_r = radius_unit_vector(xyz_max, n_xyz)
        length(Sx, Sz, cap=0.025)
        arrow_field(xyz_max, n_xyz, Sx, Sz,
                    cfunc=np.array([Sx, Sz]) * e_r, cmap='hot')
        anim.axes(x_label=r'$x/$m', y_label=r'$z/$m')
        anim.save_frame(t=t, loc='S', pos=pos)

        pos += 1

    anim.save_anim(t=t, loc='E')
    anim.save_anim(t=t, loc='H')
    anim.save_anim(t=t, loc='S')

    sys.exit(0)
