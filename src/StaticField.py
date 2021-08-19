import numpy as np
from FieldObject import Charge
from FieldPlot import static_field, static_field3d

# charge and its position
q = x = y = 1.
# quadrupole
quadrupole = []
quadrupole.append(Charge(-q, -x, y))
quadrupole.append(Charge(q, x, y))
quadrupole.append(Charge(q, -x, -y))
quadrupole.append(Charge(-q, x, -y))

# charge and x-coords of position and velocity
q = 1.
r_x = v_x = .0
# conductor loop
loop = []
for angle in np.linspace(.0, 2. * np.pi, 32, endpoint=False):
    r_y = np.cos(angle)
    r_z = np.sin(angle)
    v_y = -np.sin(angle)
    v_z = np.cos(angle)
    loop.append(Charge(1., r_x, r_y, r_z, v_x, v_y, v_z))


if __name__ == "__main__":
    xy_max = 5.
    static_field(xy_max, quadrupole, function='phi', nabla='grad')
    static_field(xy_max, loop, function='A', nabla='rot')

    xyz_max = 2.
    static_field3d(xyz_max, loop, function='B')
