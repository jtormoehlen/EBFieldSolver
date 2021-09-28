import numpy as np
from FieldObject import Charge, Current, Antenna
from FieldPlot import static_field, static_field3d

"""electrical quadrupole"""
q = x = y = 1
quadrupole = []
quadrupole.append(Charge(-q, -x, y))
quadrupole.append(Charge(q, x, y))
quadrupole.append(Charge(q, -x, -y))
quadrupole.append(Charge(-q, x, -y))

"""conductor loop"""
I = 1.0
r0_x = dl_x = 0
r0 = []
dl = []
a = 1.0
b = 2.0
for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    r0_y = a * np.cos(angle)
    r0_z = b * np.sin(angle)
    dl_y = -a * np.sin(angle)
    dl_z = b * np.cos(angle)
    r0.append(np.array([r0_x, r0_y, r0_z]))
    dl.append(np.array([dl_x, dl_y, dl_z]))
current = Current(I, r0, dl)

"""linear antenna"""
antenna = Antenna(500.0e6, 2.0)

if __name__ == "__main__":
    xy_max = 5
    # static_field(xy_max, quadrupole, function='E')
    # static_field(xy_max, quadrupole, function='phi', nabla='grad')
    # static_field(xy_max, current, function='B')
    # static_field(xy_max, current, function='A', nabla='rot')

    xz_max = 2 * antenna.lambda_0
    static_field(xz_max, antenna, function='E', t=0)
    # print(antenna.k_0 * antenna.h / np.pi)

    xyz_max = 2
    # static_field3d(xyz_max, current, function='A', nabla='rot')
    # static_field3d(xy_max, Charge(q, 0, 0, 0), function='phi', nabla='grad')
