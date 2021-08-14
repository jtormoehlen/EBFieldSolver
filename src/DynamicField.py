import sys

import numpy as np
from FieldAnimation import save_frame, save_anim
from FieldObject import Antenna
from FieldCalculator import fieldXY, fieldXZ, phi_unit, field_round, radius_unit
from FieldPlot import arrow_field

# antenna with f=500MHz and P=1W
frequency = 500.e6
power = 1.

antenna = Antenna(frequency, 1., 1./2.)
antennas = []
antennas.append(antenna)


if __name__ == "__main__":
    xyz_max = 2 * antenna.wavelength
    n_xyz = 30

    n_t = 50
    t_max = antenna.T
    t = np.linspace(0, t_max, n_t)

    for t_i in t:
        Ex, Ey, Ez = fieldXZ(xyz_max, n_xyz, antennas, t=t_i, function='E')
        arrow_field(xyz_max, Ex, Ez, cfunc=Ez)
        save_frame('E')

        Hx, Hy, Hz = fieldXY(xyz_max, n_xyz, antennas, t=t_i, function='H')
        e_phi = phi_unit(xyz_max, n_xyz)
        arrow_field(xyz_max, Hx, Hy, cfunc=Hx * e_phi)
        save_frame('H')

        Sx, Sy, Sz = fieldXZ(xyz_max, n_xyz, antennas, t=t_i, function='S')
        e_r = radius_unit(xyz_max, n_xyz)
        arrow_field(xyz_max, Sx, Sz, cfunc=Sx * e_r)
        save_frame('S')

    # render_anim('E', t)
    save_anim('E', t)
    save_anim('H', t)
    save_anim('S', t)

    sys.exit(0)
