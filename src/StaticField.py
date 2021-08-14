import numpy as np
from FieldObject import Charge
from FieldCalculator import static_field_2d, static_field_3d

# quadrupole
quadrupole = []
quadrupole.append(Charge(-1., -1., 1.))
quadrupole.append(Charge(1., 1., 1.))
quadrupole.append(Charge(1., -1., -1.))
quadrupole.append(Charge(-1., 1., -1.))

# conductor loop
loop = []
for angle in np.linspace(0, 2. * np.pi, 36):
    r_y = np.cos(angle)
    r_z = np.sin(angle)
    v_y = -np.sin(angle)
    v_z = np.cos(angle)
    loop.append(Charge(1., 0., r_y, r_z, 0., v_y, v_z))


if __name__ == "__main__":
    xy_max = 5.
    static_field_2d(xy_max, quadrupole, function='E')
    static_field_2d(xy_max, loop, function='B')

    xyz_max = 2.
    static_field_3d(xyz_max, loop, function='B')
