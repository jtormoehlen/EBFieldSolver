import sys
import FieldAnimation as anim
from FieldUtil import field, field_lines, potential_lines, forms
from FieldObject import Charge, Conductor

"initialize charges with location and q"
charges = []

# quadrupole
charges.append(Charge(-1.0, [-1.0, 1.0]))
charges.append(Charge(1.0, [1.0, 1.0]))
charges.append(Charge(1.0, [-1.0, -1.0]))
charges.append(Charge(-1.0, [1.0, -1.0]))

"initialize conductors with location and amperage"
conductors = []

# conductor loop
conductors.append(Conductor(1.0, [0.0, 3.0]))
conductors.append(Conductor(-1.0, [0.0, -3.0]))


if __name__ == "__main__":
    xy_max = 7.5
    phi, phiy, phiz = field(xy_max, xy_max, field_objects=charges, function='phi_field')
    grad_phix, grad_phiy, grad_phiz = field(xy_max, xy_max, field_objects=charges, function='phi_field', nabla='gradient')
    field_lines(xy_max, xy_max, -grad_phix, -grad_phiy)
    potential_lines(xy_max, xy_max, phi)
    forms(charges)
    anim.axes()
    anim.save_frame(loc='charges')

    xy_max = 10.
    Ax, Ay, A = field(xy_max, xy_max, field_objects=conductors, function='A_field')
    rot_Ax, rot_Ay, rot_Az = field(xy_max, xy_max, field_objects=conductors, function='A_field', nabla='curl')
    field_lines(xy_max, xy_max, rot_Ax, rot_Ay)
    potential_lines(xy_max, xy_max, A)
    forms(conductors)
    anim.axes()
    anim.save_frame(loc='conductors')

    sys.exit(0)
