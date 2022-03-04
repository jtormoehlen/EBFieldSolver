import numpy as np
from lib.FieldAnimation import static_field, static_field3d, dynamic_field
from lib.FieldObject import Charge, Current, Antenna


def main():
    """example: electrical quadrupole"""
    q = x = y = 1  # charge amount and position comps
    charges = [Charge(-q, -x, y),  # -q at (-1,1)
               Charge(q, x, y),  # q at (1,1)
               Charge(q, -x, -y),  # q at (-1,-1)
               Charge(-q, x, -y)]  # -q at (1,-1)

    """example: elliptical conductor loop"""
    I = 1.  # current
    r = []  # current elements r_i=(x_i,y_i,z_i)
    dr = []  # current element direction dr_i=(dr_x_i,dr_y_i,dr_z_i)
    a = 0.5  # semi-minor axis
    b = 1.5  # semi-major axis
    N = 20  # approximation order
    phi_n = np.linspace(0., 2 * np.pi, N, endpoint=False)  # discrete angle array
    dphi = 2 * np.pi / N  # length of dr
    for phi in phi_n:  # for each discrete angle
        r_x = dr_x = 0.  # loop in (y, z)-plane
        r_y = a * np.cos(phi)
        r_z = b * np.sin(phi)
        dr_y = -a * np.sin(phi) * dphi
        dr_z = b * np.cos(phi) * dphi
        r.append(np.array([r_x, r_y, r_z]))
        dr.append(np.array([dr_x, dr_y, dr_z]))
    currents = [Current(I, r, dr)]

    """example: linear antenna"""
    f = 1.0E9  # frequency f=1GHz
    P = 1.0  # radiation power P=1W
    n_lambda = 0.5  # multiplier so that antenna length d=n_lambda*lambda_0
    a0 = Antenna(f, P)
    l0 = a0.lambda0
    T = a0.T

    # static_field([-5., 5., -5., 5.], charges, nabla='grad', ffunc='phi')
    # static_field([-5., 5., -5., 5.], currents, nabla='rot', ffunc='A')

    # static_field3d([-5., 5., -5., 5., -5., 5.], charges, ffunc='E')
    # static_field3d([-5., 5., -5., 5., -5., 5.], currents, nabla='rot', ffunc='A')

    dynamic_field([-2 * l0, 2 * l0, -2 * l0, 2 * l0], T, [a0], ffunc='E', save=True)


if __name__ == "__main__":
    main()
