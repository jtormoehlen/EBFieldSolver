import sys
import numpy as np
from matplotlib import pyplot as plt

from FieldAnimation import axes, save_frame
from FieldUtil import field, field_lines, potential_lines, forms, field3d, arrow_field3d
from FieldObject import Charge, Current

"initialize charges with location and q"
charges = []

# quadrupole
charges.append(Charge(-1., [-1., 1., 0.]))
charges.append(Charge(1., [1., 1., 0.]))
charges.append(Charge(1., [-1., -1., 0.]))
charges.append(Charge(-1., [1., -1., 0.]))

"initialize electric circulating current with location, charge and velocity"
currents = []
for angle in np.linspace(0, 2. * np.pi, 8):
    r_z = np.cos(angle)
    r_y = np.sin(angle)
    v_z = -np.sin(angle)
    v_y = np.cos(angle)
    currents.append(Current(1., [0., r_y, r_z], [0., v_y, v_z]))


if __name__ == "__main__":
    xy_max = 5.
    n_xy = 30.
    phi, phiy, phiz = field(xy_max, n_xy, objects=charges, function='phi')
    Ex, Ey, Ez = field(xy_max, n_xy, objects=charges, nabla='gradient', function='phi')
    field_lines(xy_max, n_xy, -Ex, -Ey)
    potential_lines(xy_max, n_xy, phi)
    forms(charges)
    axes()
    save_frame(loc='charges')

    xy_max = 7.5
    n_xy = 30.
    Bx, By, Bz = field(xy_max, n_xy, objects=currents, function='B')
    field_lines(xy_max, n_xy, Bx, By)
    axes()
    save_frame(loc='circ_current2d')

    xy_max = 2.
    n_xy = 6.
    Bx, By, Bz = field3d(xy_max, n_xy, objects=currents, function='B')
    arrow_field3d(xy_max, n_xy, Bx, By, Bz)
    plt.quiver(0., 0., 0., 0., 0., 1., color='red')
    plt.quiver(0., 0., 1., 0., -1., 0., color='green')
    axes()
    save_frame(loc='circ_current3d')

    sys.exit(0)
