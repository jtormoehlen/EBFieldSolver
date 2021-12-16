import numpy as np
from FieldObject import Charge, Current
from FieldAnimation import static_field, static_field3d

# electrical quadrupole
q = x = y = 1
quadrupole = []
quadrupole.append(Charge(-q, -x, y))
quadrupole.append(Charge(q, x, y))
quadrupole.append(Charge(q, -x, -y))
quadrupole.append(Charge(-q, x, -y))

# conductor loop
I = 1.
r_x = dr_x = 0.
r = []
dr = []
a = 0.5
b = 1.5
N = 20
phi_n = np.linspace(0., 2 * np.pi, N, endpoint=False)
dphi = 2 * np.pi / N
for phi in phi_n:
    r_y = a * np.cos(phi)
    r_z = b * np.sin(phi)
    dr_y = -a * np.sin(phi) * dphi
    dr_z = b * np.cos(phi) * dphi
    r.append(np.array([r_x, r_y, r_z]))
    dr.append(np.array([dr_x, dr_y, dr_z]))
current = Current(I, r, dr)

if __name__ == "__main__":
    xy_max = 5
    # static_field(xy_max, quadrupole, function='E')
    # static_field(xy_max, quadrupole, nabla='grad', function='phi')
    # static_field(xy_max, current, function='B')
    static_field(xy_max, current, nabla='rot', function='A')

    xyz_max = 2.
    # static_field3d(xyz_max, current, nabla='rot', function='A')
    # static_field3d(xy_max, quadrupole, nabla='grad', function='phi')
