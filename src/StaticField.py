import numpy as np
from FieldCalculator import fieldXY, fieldXZ, field3d
from FieldPlot import field_lines, potential_lines, arrow_field3d
from FieldObject import Charge

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
    n_xy = 20.
    phi, phiy, phiz = fieldXY(xy_max, n_xy, quadrupole, function='phi')
    Ex, Ey, Ez = fieldXY(xy_max, n_xy, quadrupole, nabla='gradient', function='phi')
    potential_lines(xy_max, phi)
    field_lines(xy_max, -Ex, -Ey, quadrupole)

    Bx, B, Bz = fieldXZ(xy_max, n_xy, loop, function='B')
    field_lines(xy_max, Bx, Bz, loop)

    xyz_max = 2.
    n_xyz = 6.
    Bx, By, Bz = field3d(xyz_max, n_xyz, loop, function='B')
    arrow_field3d(xyz_max, Bx, By, Bz)
