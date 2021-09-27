import numpy as np
from FieldObject import Charge, Current, Antenna
from FieldPlot import static_field, static_field3d

# quadrupole
q = x = y = 1.
quadrupole = []
quadrupole.append(Charge(-q, -x, y))
quadrupole.append(Charge(q, x, y))
quadrupole.append(Charge(q, -x, -y))
quadrupole.append(Charge(-q, x, -y))

charge_pos = Charge(q, 1., 0.)
charge_neg = Charge(-q, -1., 0.)
dipole = []
dipole.append(charge_neg)
dipole.append(charge_pos)

# conductor loop
I = 1.
r0_y = dl_y = .0
r0 = []
dl = []
a = 0.001
b = 0.001
for angle in np.linspace(.0, 2. * np.pi, 10, endpoint=False):
    r0_x = a * np.cos(angle)
    r0_z = b * np.sin(angle)
    dl_x = -a * np.sin(angle)
    dl_z = b * np.cos(angle)
    r0.append(np.array([r0_x, r0_y, r0_z]))
    dl.append(np.array([dl_x, dl_y, dl_z]))
current = Current(I, r0, dl)

antenna = Antenna(500e6, 1.0, 3./2.)

if __name__ == "__main__":
    xy_max = 5.
    # static_field(xy_max, quadrupole, function='E')
    # static_field(xy_max, current, function='A', nabla='rot')
    # static_field(xy_max, dipole, function='phi', nabla='grad')
    # static_field(xy_max, dipole, function='phi', nabla='grad')
    # static_field(2. * antenna.lambda_0, antenna, function='E')
    print(antenna.k_0 * antenna.h / np.pi)

    xyz_max = 2.
    # static_field3d(xyz_max, current, function='A', nabla='rot')
    # static_field3d(xy_max, Charge(q, .0, .0, .0), function='phi', nabla='grad')
