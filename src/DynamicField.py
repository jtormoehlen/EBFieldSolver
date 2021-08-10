import sys
import numpy as np
from FieldAnimation import show_frame, save_frame, save_anim
from FieldObject import Antenna
from FieldUtil import field, arrow_field, phi_unit, field_round, radius_unit

r0 = [0., 0., 0.]
frequency = 500.E6
power = 1.

antenna = Antenna(frequency, 1., 1./2.)
antennae = []
antennae.append(antenna)


if __name__ == "__main__":
    xyz_max = 2 * antenna.wavelength
    n_xyz = 30

    n_t = 50
    t_max = antenna.T
    t = np.linspace(0, t_max, n_t)

    index = 0
    for t_i in t:
        Ex, Ey, Ez = field(xyz_max, n_xyz, t=t_i, plane='xz', objects=antennae, function='E')
        Hx, Hy, Hz = field(xyz_max, n_xyz, t=t_i, plane='xy', objects=antennae, function='H')
        Sx, Sy, Sz = field(xyz_max, n_xyz, t=t_i, plane='xz', objects=antennae, function='S')

        field_round(Ex, Ez, xyz_max, n_xyz, 5.)
        arrow_field(xyz_max, n_xyz, Ex, Ez,
                    cfunc=Ez)
        show_frame(y_label='$z$', location='E')
        save_frame(t=t, location='E', index=index)

        e_phi = phi_unit(xyz_max, n_xyz)
        field_round(Hx, Hy, xyz_max, n_xyz, 1.e-2)
        arrow_field(xyz_max, n_xyz, Hx, Hy,
                    cfunc=Hx * e_phi)
        show_frame(location='H')
        save_frame(t=t, location='H', index=index)

        e_r = radius_unit(xyz_max, n_xyz)
        field_round(Sx, Sz, xyz_max, n_xyz, 1.e-3)
        arrow_field(xyz_max, n_xyz, Sx, Sz,
                    cfunc=Sx * e_r)
        show_frame(y_label='$z$', location='S')
        save_frame(t=t, location='S', index=index)

        index += 1

    save_anim(t=t, location='E')
    save_anim(t=t, location='H')
    save_anim(t=t, location='S')

    sys.exit(0)
