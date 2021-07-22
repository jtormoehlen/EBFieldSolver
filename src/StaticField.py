import sys

import matplotlib.pyplot as plt
import numpy as np
import FieldUtil as util
import FieldAnimation as anim
import FieldOperation

from FieldObject import PointCharge, Conductor

"initialize charges with location and q"
charges = []

# dy = np.arange(-4.0, 4.0, 0.05)
# for i in dy:
#     charges.append(PointCharge(10.0, [-1.0, i]))
#
# for i in dy:
#     charges.append(PointCharge(-10.0, [1.0, i]))

# dipole charge
# charges.append(PointCharge(1.0, [0.0, -1.0]))
# charges.append(PointCharge(-1.0, [0.0, 1.0]))

# quadrupole charge
charges.append(PointCharge(-1.0, [-1.0, 1.0]))
charges.append(PointCharge(1.0, [1.0, 1.0]))
charges.append(PointCharge(1.0, [-1.0, -1.0]))
charges.append(PointCharge(-1.0, [1.0, -1.0]))

# single positive charge
# charges.append(PointCharge(1.0, [0.0, 0.0]))

"initialize conductors with amperage and q"
conductors = []
# single wire
# conductors.append(Conductor(1.0, [0.0, 0.0]))

# conductor loop
conductors.append(Conductor(-1.0, [0.0, 3.0]))
conductors.append(Conductor(1.0, [0.0, -3.0]))

# coil
# for i in np.linspace(-5, 5, 10):
#     conductors.append(Conductor(-1.0, [i - 0.5, 3.0]))
#     conductors.append(Conductor(1.0, [i, -3.0]))


def compute_total_field(x, y, field_objects, potential_field=False):
    x_field, y_field = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
    for field_object in field_objects:
        for i in range(len(x)):
            for j in range(len(y)):
                if potential_field:
                    field = field_object.compute_potential(x[i][j], y[i][j])
                    x_field[i][j] += field
                else:
                    field = field_object.compute_field(x[i][j], y[i][j])
                    x_field[i][j] += field[0]
                    y_field[i][j] += field[1]
    return [x_field, y_field]


def norm_total_field(x, y, field_objects):
    field_x, field_y = compute_total_field(x, y, field_objects)
    return np.hypot(field_x, field_y)


def compute_bodies(field_objects):
    bodies = []
    for field_object in field_objects:
        bodies.append(field_object.body())
    return bodies


def compute_details(field_objects):
    details = []
    for field_object in field_objects:
        details.append(field_object.details())
    return details


if __name__ == "__main__":
    n_xy = 30
    xy_max = 7.5
    X, Y = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                       np.linspace(-xy_max, xy_max, n_xy))

    E = compute_total_field(X, Y, charges)
    Z = compute_total_field(X, Y, charges, potential_field=True)
    levels = np.linspace(np.min(Z[0]) / 10, np.max(Z[0]) / 10, 10)
    total_bodies = compute_bodies(charges)

    div_E = FieldOperation.divergence(E)
    grad_phi = FieldOperation.gradient(Z[0])
    grad_phi_norm = np.hypot(grad_phi[0], grad_phi[1])

    # util.plot_arrows(X, Y, grad_phi[0] / grad_phi_norm, grad_phi[1] / grad_phi_norm)
    # util.plot_intensity(X, Y, div_E, cmap='coolwarm')
    util.plot_streamlines(X, Y, grad_phi[0], grad_phi[1], color='grey')
    util.plot_contourf(X, Y, Z[0], levels)
    util.plot_forms(total_bodies)

    anim.render_frame(loc='charges', aspect=True)

    B = compute_total_field(X, Y, conductors)
    total_bodies = compute_bodies(conductors)
    total_details = compute_details(conductors)

    util.plot_streamlines(X, Y, B[0], B[1], color=np.log(np.hypot(B[0], B[1])), cmap='cool', zorder=1, density=2)
    util.plot_forms(total_bodies)
    util.plot_details(total_details)

    anim.render_frame(loc='conductors', aspect=True)

    sys.exit(0)
