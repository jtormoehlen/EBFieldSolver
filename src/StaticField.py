import sys

import numpy as np
import FieldUtil as util
import FieldAnimation as anim

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
# charges.append(PointCharge(-1.0, [0.0, 0.0]))

"initialize conductors with amperage and q"
conductors = []
# single wire
# conductors.append(Conductor(50E-3, [0.0, 0.0]))

# conductor loop
conductors.append(Conductor(-1.0, [0.0, 3.0]))
conductors.append(Conductor(1.0, [0.0, -3.0]))

# coil
# for i in np.linspace(-5, 5, 10):
#     conductors.append(Conductor(-1.0, [i - 0.5, 3.0]))
#     conductors.append(Conductor(1.0, [i, -3.0]))


def compute_total_field(x, y, field_objects, potential_field=False):
    fields = []
    for field_object in field_objects:
        if potential_field:
            fields.append(field_object.compute_potential(x, y))
        else:
            fields.append(field_object.compute_field(x, y))

    total_field = np.zeros_like(fields[0])
    for field in fields:
        total_field += field
    return total_field


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
    n_xy = 100
    xy_max = 10
    X, Y = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                       np.linspace(-xy_max, xy_max, n_xy))

    Ex, Ey = compute_total_field(X, Y, charges)
    Z = compute_total_field(X, Y, charges, potential_field=True)
    levels = np.linspace(np.min(Z) / 10, np.max(Z) / 10, 10)
    total_bodies = compute_bodies(charges)

    util.plot_streamlines(X, Y, Ex, Ey, color='grey')
    util.plot_contourf(X, Y, Z, levels)
    util.plot_forms(total_bodies)

    anim.render_frame(loc='quadrupol', aspect=True)

    Bx, By = compute_total_field(X, Y, conductors)
    total_bodies = compute_bodies(conductors)
    total_details = compute_details(conductors)

    util.plot_streamlines(X, Y, Bx, By, color=np.log(np.hypot(Bx, By)), cmap='cool', zorder=1, density=2)
    util.plot_forms(total_bodies)
    util.plot_details(total_details)

    anim.render_frame(loc='wireloop', aspect=True)

    sys.exit(0)
