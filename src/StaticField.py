import sys
import numpy as np
import matplotlib.pyplot as plt
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
    Z = np.zeros((len(X), len(X)))

    phi, phiy, phiz = util.total_field(X, Y, Z, charges, f='phi_field')
    phi_levels = np.linspace(np.min(phi) / 10, np.max(phi) / 10, 4)
    grad_phix, grad_phiy, grad_phiz = util.total_diff(X, Y, Z, charges, f='phi_field', nabla='gradient')
    grad_phi_norm = np.hypot(grad_phix, grad_phiy)

    plt.streamplot(X, Y, -grad_phix, -grad_phiy, color=np.log(grad_phi_norm), cmap='cool')
    plt.contour(X, Y, phi, phi_levels, colors='k', alpha=0.5)
    util.forms(charges)
    anim.window()
    anim.render_frame(loc='charges')

    Ax, Ay, A = util.total_field(X, Y, Z, conductors, f='A_field')
    A_levels = np.linspace(np.min(A), np.max(A), 7)
    rot_Ax, rot_Ay, rot_Az = util.total_diff(X, Y, Z, conductors, f='A_field', nabla='curl')
    rot_A_norm = np.hypot(rot_Ax, rot_Ay)

    plt.streamplot(X, Y, rot_Ax, rot_Ay, color=np.log(rot_A_norm), cmap='cool')
    plt.contour(X, Y, A, A_levels, colors='k', alpha=0.5)
    util.forms(conductors)
    util.details(conductors)
    anim.window()
    anim.render_frame(loc='conductors')

    sys.exit(0)
