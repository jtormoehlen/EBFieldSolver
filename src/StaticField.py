import numpy as np
from FieldObject import Charge, Current
from FieldPlot import static_field, static_field3d

# electrical quadrupole
q = x = y = 1
quadrupole = []
quadrupole.append(Charge(-q, -x, y))
quadrupole.append(Charge(q, x, y))
quadrupole.append(Charge(q, -x, -y))
quadrupole.append(Charge(-q, x, -y))

# conductor loop
I = 1.0
r0_x = dl_x = 0
r0 = []
dl = []
a = b = 0.1
# N = 10
# phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
# delta_phi = 2 * np.pi / N
for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    r0_y = a * np.cos(angle)
    r0_z = b * np.sin(angle)
    dl_y = -a * np.sin(angle)
    dl_z = b * np.cos(angle)
    r0.append(np.array([r0_x, r0_y, r0_z]))
    dl.append(np.array([dl_x, dl_y, dl_z]))
current = Current(I, r0, dl)

if __name__ == "__main__":
    xy_max = 5
    # static_field(xy_max, quadrupole, function='E')
    # static_field(xy_max, quadrupole, nabla='grad', function='phi')
    # static_field(xy_max, current, function='B')
    static_field(xy_max, current, nabla='rot', function='A')

    xyz_max = 2
    # static_field3d(xyz_max, current, nabla='rot', function='A')
    static_field3d(xy_max, Charge(q, 0, 0, 0), nabla='grad', function='phi')
