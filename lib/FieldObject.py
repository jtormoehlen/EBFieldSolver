import numpy as np

c = 2.998E10  # speed of light in cm/s


class Charge:
    def __init__(self, q, x, y, z=0):
        """
        Point charge in gaussian cgs-units.
        :param q: charge
        :param x: position x
        :param y: position y
        :param z: position z
        """
        self.q = q
        self.r0 = np.array([x, y, z])

    def E(self, x, y, z, t=0):
        r = np.array([x, y, z])
        return self.q * (r - self.r0) / np.linalg.norm(r - self.r0) ** 3

    def phi(self, x, y, z, t=0):
        r = np.array([x, y, z])
        return self.q / np.linalg.norm(r - self.r0) * np.array([1., 0., 0.])


class Current:
    def __init__(self, I, r0, dr):
        """
        Current loop in gaussian cgs-units.
        :param I: current
        :param r0: list of positions [(x_1,y_1,z_1),...,
                                      (x_n-1,y_n-1,z_n-1)]
        :param dr: list of current elements [(drx_1,dry_1,drz_1),...,
                                             (drx_n-1,dry_n-1,drz_n-1)]
        """
        self.I = I
        self.r0 = r0
        self.dr = dr

    def B(self, x, y, z, t=0):
        r = np.array([x, y, z])
        B = 0
        for i in range(len(self.r0)):
            dl_cross_r_r0 = np.cross(self.dr[i], r - self.r0[i])
            B += (1 / c) * self.I * dl_cross_r_r0 / np.linalg.norm(r - self.r0[i]) ** 3
        return B

    def A(self, x, y, z, t=0):
        r = np.array([x, y, z])
        A = 0
        for i in range(len(self.r0)):
            A += (1 / c) * self.I * self.dr[i] / np.linalg.norm(r - self.r0[i])
        return A


class Antenna:
    def __init__(self, f, P, d_fac=0, x=0, y=0, z=0):
        """
        Antenna in gaussian cgs-units.
        :param f: frequency
        :param P: average radiation power
        :param d_fac: antenna length factor d=l_fac*l
        :param x: position x
        :param y: position y
        :param z: position z
        """
        self.r0 = np.array([x, y, z])
        self.P = P * 1.0E7
        self.omega = 2 * np.pi * f
        self.l = c / f
        self.k = (2 * np.pi) / self.l
        self.d_fac = d_fac
        self.d = d_fac * self.l

    def E(self, x, y, z, t=0):
        return self.A(x, y, z, t) * (1j / self.k)

    def B(self, x, y, z, t=0):
        return self.A(x, y, z, t)

    def A(self, x, y, z, t=0):
        r = np.sqrt((x - self.r0[0]) ** 2 + (y - self.r0[1]) ** 2 + (z - self.r0[2]) ** 2)
        f_rt = np.exp(1j * (self.k * r - self.omega * t))
        I = np.sqrt(c * self.P)
        if self.d == 0:  # short dipole
            I *= np.sqrt(3.0)
            p = 1j * (c * I) / self.omega ** 2 * np.array([0, 0, 1])
            return -1j * self.k * p * (f_rt / r)
        else:  # linear antenna
            theta = np.arccos(z / r)
            f_theta = (np.cos(self.k * self.d / 2 * np.cos(theta)) -
                       np.cos(self.k * self.d / 2)) / np.sin(theta) ** 2
            if not self.d_fac.is_integer():
                I *= np.sqrt(1 / 1.2188)
            else:
                I *= np.sqrt(1 / 3.3181)
            return (2 * I / self.omega) * (f_rt / r) * f_theta * np.array([0, 0, 1])
