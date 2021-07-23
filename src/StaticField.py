import sys

import numpy as np
import FieldUtil as util
import FieldAnimation as anim
import FieldOperator as fo

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

"initialize conductors with location and amperage"
conductors = []
# single wire
# conductors.append(Conductor(1.0, [0.0, 0.0]))

# conductor loop
conductors.append(Conductor(1.0, [0.0, 3.0]))
conductors.append(Conductor(-1.0, [0.0, -3.0]))

# coil
# for i in np.linspace(-5, 5, 10):
#     conductors.append(Conductor(1.0, [i - 0.5, 3.0]))
#     conductors.append(Conductor(-1.0, [i, -3.0]))


if __name__ == "__main__":
    n_xy = 30
    xy_max = 7.5
    X, Y = np.meshgrid(np.linspace(-xy_max, xy_max, n_xy),
                       np.linspace(-xy_max, xy_max, n_xy))

    phi, phiy, phiz = util.compute_total_field(X, Y, charges)
    phi_levels = np.linspace(np.min(phi) / 10, np.max(phi) / 10, 4)
    Ax, Ay, A = util.compute_total_field(X, Y, conductors)

    grad_phix, grad_phiy = np.zeros((len(X), len(Y))), np.zeros((len(X), len(Y)))
    rot_Ax, rot_Ay = np.zeros((len(X), len(Y))), np.zeros((len(X), len(Y)))
    div_E = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            for charge in charges:
                grad_x, grad_y, grad_z = fo.gradient(X[i][j], Y[i][j], 0, charge.compute_potential)
                grad_phix[i][j] -= grad_x
                grad_phiy[i][j] -= grad_y
                div_E[i][j] = fo.divergence(X[i][j], Y[i][j], 0, charge.compute_field)
            for conductor in conductors:
                rot_x, rot_y, rot_z = fo.curl(X[i][j], Y[i][j], 0, conductor.compute_potential)
                rot_Ax[i][j] += rot_x
                rot_Ay[i][j] += rot_y

    grad_phi_norm = np.hypot(grad_phix, grad_phiy)
    rot_A_norm = np.hypot(rot_Ax, rot_Ay)

    E_total_forms = util.compute_forms(charges)
    B_total_forms = util.compute_forms(conductors)
    B_total_details = util.compute_details(conductors)

    util.plot_streamlines(X, Y, grad_phix, grad_phiy, color=np.log(grad_phi_norm), cmap='cool')
    util.plot_contour(X, Y, phi, phi_levels)
    util.plot_forms(E_total_forms)
    anim.window()
    anim.render_frame(loc='charges')

    util.plot_streamlines(X, Y, rot_Ax, rot_Ay, color=np.log(rot_A_norm), cmap='cool')
    util.plot_contour(X, Y, A)
    util.plot_forms(B_total_forms)
    util.plot_details(B_total_details)
    anim.window()
    anim.render_frame(loc='conductors')

    sys.exit(0)
