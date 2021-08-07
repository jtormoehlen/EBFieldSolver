import sys
import numpy as np
import FieldAnimation as anim
from FieldObject import HertzDipole, DipoleAntenna
from FieldUtil import field, arrow_field, phi_unit_vector, radius_unit_vector, field_round

r0 = [0., 0., 0.]
frequency = 500.E6
power = 1.
dipole = []
dipole.append(HertzDipole(r0, frequency, power))
# dipole.append(DipoleAntenna(r0, frequency, 1., 0.5))


if __name__ == "__main__":
    xyz_max = 2 * dipole[0].wavelength
    n_xyz = 30

    n_t = 50
    t_max = dipole[0].T
    t = np.linspace(0, t_max, n_t)

    index = 0
    for t_i in t:
        Ex, Ey, Ez = field(xyz_max, n_xyz, plane='xz', objects=dipole, function='E', t=t_i)
        Hx, Hy, Hz = field(xyz_max, n_xyz, plane='xy', objects=dipole, function='H', t=t_i)
        Sx, Sy, Sz = field(xyz_max, n_xyz, plane='xz', objects=dipole, function='S', t=t_i)

        field_round(Ex, Ez, xyz_max, n_xyz, 50.)
        arrow_field(xyz_max, n_xyz, Ex, Ez,
                    cfunc=Ez, cmap='winter')
        anim.show_frame(y_label='$z$', location='E', back_color='black')
        anim.save_frame(t=t, location='E', index=index)

        e_phi = phi_unit_vector(xyz_max, n_xyz)
        field_round(Hx, Hy, xyz_max, n_xyz, 0.5)
        arrow_field(xyz_max, n_xyz, Hx, Hy,
                    cfunc=Hx * e_phi, cmap='cool')
        anim.show_frame(location='H', back_color='black')
        anim.save_frame(t=t, location='H', index=index)

        e_r = radius_unit_vector(xyz_max, n_xyz)
        field_round(Sx, Sz, xyz_max, n_xyz, 0.5)
        arrow_field(xyz_max, n_xyz, Sx, Sz,
                    cfunc=Sx * e_r, cmap='hot')
        anim.show_frame(y_label='$z$', location='S', back_color='black')
        anim.save_frame(t=t, location='S', index=index)

        index += 1

    anim.save_anim(t=t, location='E')
    anim.save_anim(t=t, location='H')
    anim.save_anim(t=t, location='S')

    sys.exit(0)
