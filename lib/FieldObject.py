import numpy as np

# constants
epsilon_0 = 1.
mu_0 = 1.
c = 299792458.
Z_0 = np.sqrt(mu_0 / epsilon_0)


class Charge:
    def __init__(self, q, x, y, z=0):
        """
        Point charge.
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
        return self.q / np.linalg.norm(r - self.r0) * np.array([1, 0, 0])


class Current:
    def __init__(self, I, r0, dr):
        """
        Current loop.
        :param I: Amperage
        :param r0: position (x, y, z)
        :param dr: current element
        """
        self.I = I
        self.r0 = r0
        self.dr = dr

    def B(self, x, y, z, t=0):
        r = np.array([x, y, z])
        B = 0
        for i in range(len(self.r0)):
            dl_cross_r_r0 = np.cross(self.dr[i], r - self.r0[i])
            B += self.I * dl_cross_r_r0 / np.linalg.norm(r - self.r0[i]) ** 3
        return B

    def A(self, x, y, z, t=0):
        r = np.array([x, y, z])
        A = 0
        for i in range(len(self.r0)):
            A += self.I * self.dr[i] / np.linalg.norm(r - self.r0[i])
        return A


class Antenna:
    def __init__(self, nu, P, k=0, x=0, y=0, z=0, dphi=0.):
        """
        Antenna.
        :param nu: radiation frequency f
        :param P: average radiation power P
        :param k: antenna length factor: short or linear dipole
        :param x: position x
        :param y: position y
        :param z: position z
        """
        self.r0 = np.array([x, y, z])
        self.P = P
        self.T = 1 / nu
        self.omega = 2 * np.pi * nu
        self.lambda0 = c / nu
        self.k_0 = (2 * np.pi) / self.lambda0
        self.rod = k
        self.L = k * self.lambda0
        self.h = self.L / 2
        self.dphi = np.deg2rad(dphi)

    def p(self):
        e_z = np.array([0, 0, 1])
        I_0 = np.sqrt((48 * np.pi * self.P / Z_0)) / self.k_0
        return (1j * I_0) / (2. * self.omega) * e_z

    def I(self):
        return 0

    def E(self, x, y, z, t=0):
        return self.A(x, y, z, t) * (1j * Z_0) / (mu_0 * self.k_0)

    def H(self, x, y, z, t=0):
        return self.A(x, y, z, t) * (1 / mu_0)

    def A(self, x, y, z, t=0):
        r = np.sqrt((x - self.r0[0]) ** 2 + (y - self.r0[1]) ** 2 + (z - self.r0[2]) ** 2)
        if self.L == 0:
            p = self.p()
            const = -(1j * mu_0 * self.omega) / (4 * np.pi)
            return const * p * ((np.exp(1j * (self.k_0 * r - self.omega * t - self.dphi))) / r)
        else:
            theta = np.arccos(z / r)
            f_theta_phi = (np.cos(self.k_0 * self.h * np.cos(theta)) - np.cos(self.k_0 * self.h)) / np.sin(theta) ** 2
            exp_t = np.exp(1j * (self.k_0 * r - self.omega * t - self.dphi))

            rod_int = self.rod.is_integer()
            if not rod_int:
                I_0 = np.sqrt((4 * np.pi * self.P) / (Z_0 * 1.21883))
            else:
                I_0 = np.sqrt((4 * np.pi * self.P) / (Z_0 * 3.31813))
            A_z = ((mu_0 * I_0 * exp_t) / (2 * np.pi * self.k_0 * r)) * f_theta_phi
            return np.array([0, 0, A_z])
