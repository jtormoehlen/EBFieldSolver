import numpy as np
from matplotlib import pyplot as plt

from lib.FieldAnimation import static_field, static_field3d, dynamic_field
from lib.FieldObject import Charge, Current, Antenna


def main():
    # ++++++example: electrical quadrupole++++++
    q = 1.0  # charge in Q
    x = y = 1  # x,y in a
    chargeList = [Charge(-q, -x, y),  # -q at (-1,1)
                  Charge(q, x, y),  # q at (1,1)
                  Charge(q, -x, -y),  # q at (-1,-1)
                  Charge(-q, x, -y)]  # -q at (1,-1)

    # ++++++example: elliptical conductor loop++++++
    I = 1.0  # current in I_0
    r = []  # current elements positions r_i=(x_i,y_i,z_i)
    dr = []  # current element directions dr_i=(dr_x_i,dr_y_i,dr_z_i)
    a = 5  # semi-minor axis in cm
    b = 15  # semi-major axis in cm
    N = 50  # approximation order
    phi_n = np.linspace(0., 2 * np.pi, N, endpoint=False)  # discrete angle array
    dphi = 2 * np.pi / N  # length of dr
    for phi in phi_n:  # loop over all angles
        r_x = dr_x = 0.  # loop in (y, z)-plane
        r_y = a * np.cos(phi)
        r_z = b * np.sin(phi)
        dr_y = -a * np.sin(phi) * dphi
        dr_z = b * np.cos(phi) * dphi
        r.append(np.array([r_x, r_y, r_z]))  # add pos to list
        dr.append(np.array([dr_x, dr_y, dr_z]))  # add dir to list
    currentList = [Current(I, r, dr)]  # current list approx current loop

    # ++++++example: linear antenna++++++
    f = 2.0E9  # frequency in GHz
    P = 1.0  # radiation power in W
    l_fac = 1.0  # multiplier so that antenna length d=l_fac*lambda
    antenna = Antenna(f, P, l_fac)
    l = antenna.l  # wave length in cm
    antennaList = [antenna]
    T = 1 / f  # period

    static_field([-5., 5., -5., 5.], chargeList, nabla='grad', ffunc='phi')
    # static_field([-20., 20., -20., 20.], currentList, nabla='rot', ffunc='A')

    # static_field3d([-2., 2., -2., 2., -2., 2.], chargeList, nabla='grad', ffunc='phi')
    # static_field3d([-20., 20., -20., 20., -20., 20.], currentList, nabla='rot', ffunc='A')

    # dynamic_field([-4 * l, 4 * l, -4 * l, 4 * l], T, antennaList, ffunc='E', save=True)


if __name__ == "__main__":
    main()
