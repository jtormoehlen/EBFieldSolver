import sys
import numpy as np
from FieldAnimation import show_frame
from FieldUtil import field, field_lines, potential_lines, arrow_field, forms, field3d, arrow_field3d
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
    phi, phiy, phiz = field(xy_max, n_xy, objects=quadrupole, function='phi')
    Ex, Ey, Ez = field(xy_max, n_xy, objects=quadrupole, nabla='gradient', function='phi')
    arrow_field(xy_max, n_xy, -Ex, -Ey, normalize=True)
    potential_lines(xy_max, n_xy, phi)
    forms(quadrupole)
    show_frame()

    Bx, By, Bz = field(xy_max, n_xy, plane='xz', objects=loop, function='B')
    arrow_field(xy_max, n_xy, Bx, Bz, normalize=True)
    forms(loop)
    show_frame(y_label='$z$')

    xyz_max = 2.
    n_xyz = 6.
    Bx, By, Bz = field3d(xyz_max, n_xyz, objects=loop, function='B')
    arrow_field3d(xyz_max, n_xyz, Bx, By, Bz)

    # sys.exit(0)
