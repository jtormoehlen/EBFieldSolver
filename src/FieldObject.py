import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from FieldOperator import spherical_to_cartesian

epsilon_0 = 8.85e-12
mu_0 = 4. * np.pi * 10.e-7
c = 299792458.
Z_0 = np.sqrt(mu_0 / epsilon_0)


class Charge:
    e_const = 1 / (4 * np.pi * epsilon_0)
    m_const = (mu_0 / (4 * np.pi))

    def __init__(self, q, r0_x, r0_y, r0_z=0, v_x=0, v_y=0, v_z=0):
        self.q = q
        self.r0 = np.array([r0_x, r0_y, r0_z])
        self.v = np.array([v_x, v_y, v_z])
        self.R = 0.25

    def form(self):
        if np.linalg.norm(self.v) == 0:
            qcolor = 'blue' if self.q < 0 else 'red'
            circle = plt.Circle((self.r0[0], self.r0[1]), self.R, color=qcolor)
            plt.gca().add_patch(circle)
        else:
            isymbol = 'x' if self.v[2] < 0 else 'o'
            circle = plt.Circle((self.r0[0], self.r0[2]), self.R, color='grey')
            plt.gca().add_patch(circle)
            # plt.scatter(self.r0[0], self.r0[2], self.R*10, 'black', isymbol, zorder=2)

    def E(self, x, y, z):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        E = self.e_const * self.q * ((r - self.r0) / r_r0_norm ** 3)
        return E

    def phi(self, x, y, z):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        phi = self.e_const * self.q * (1 / r_r0_norm) * np.array([1., 0., 0.])
        return phi

    def B(self, x, y, z):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        v_cross_r_r0 = np.cross(self.v, r - self.r0)
        B = self.m_const * ((self.q * v_cross_r_r0) / (r_r0_norm ** 3))
        return B

    def A(self, x, y, z):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        A = self.m_const * ((self.q * self.v) / r_r0_norm)
        return A


class Antenna:
    def __init__(self, frequency, power, L=0.):
        self.r0 = np.array([0., 0., 0.])
        self.frequency = frequency
        self.T = 1. / frequency
        self.power = power
        self.omega = 2. * np.pi * self.frequency
        self.lambda_0 = c / self.frequency
        self.k_0 = (2. * np.pi) / self.lambda_0
        self.L = L * self.lambda_0
        self.h = L / 2.
        self.I_0 = power / 1.e3
        self.p_z = np.sqrt(12. * np.pi * c * self.power / (mu_0 * self.omega ** 4))
        self.factor = 0.

    def form(self):
        ellipse = patch.Ellipse((self.r0[0], self.r0[1]), 0.1, 0.2, color='grey', alpha=0.5)
        plt.gca().add_patch(ellipse)

    def p(self, t):
        e_z = np.array([0., 0., 1.])
        p = self.p_z * np.exp(-1j * self.omega * t) * e_z
        return p

    def far_field(self, x, y, z, t):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        f_theta_phi = (np.cos(self.k_0 * self.h * np.cos(theta)) - np.cos(self.k_0 * self.h)) / np.sin(theta)
        e_t = np.exp(1.j * (self.k_0 * r - self.omega * t))
        field = 1.j * self.I_0 * (e_t / (2 * np.pi * r)) * f_theta_phi
        return field

    def E(self, x, y, z, t):
        if self.L == 0:
            self.factor = 50.
            p = self.p(t)
            r = np.array([x, y, z]) - self.r0
            r_norm = np.linalg.norm(r)
            r = r / r_norm
            r_cross_p = np.cross(r, p)
            rcrossp_cross_r = np.cross(r_cross_p, r)
            r_dot_p = np.dot(r, p)
            r_dot_rdotp = np.dot(3 * r, r_dot_p) - p

            const = (self.omega ** 3 / (4. * np.pi * epsilon_0 * c ** 3))
            rho = (self.omega * r_norm) / c
            far = 1. / rho
            near_3 = 1. / rho ** 3
            near_2 = 1.j / rho ** 2
            e_pow = 1.j * (rho - (self.omega * t))
            E = const * ((rcrossp_cross_r * far) + (r_dot_rdotp * (near_3 - near_2))) * np.exp(e_pow)
            return E
        else:
            self.factor = 5.
            E_theta = Z_0 * self.far_field(x, y, z, t)
            E = np.array([0., E_theta, 0.])
            E_x, E_y, E_z = spherical_to_cartesian(x, y, z, E)
            return [E_x, E_y, E_z]

    def H(self, x, y, z, t):
        if self.L == 0:
            self.factor = 0.5
            p = self.p(t)
            r = np.array([x, y, z]) - self.r0
            r_norm = np.linalg.norm(r)
            r = r / r_norm
            r_cross_p = np.cross(r, p)

            const = (self.omega ** 3 / (4. * np.pi * c ** 2))
            rho = (self.omega * r_norm) / c
            far = 1. / rho
            near_2 = 1.j / rho ** 2
            e_pow = 1.j * (rho - (self.omega * t))
            H = const * r_cross_p * (far + near_2) * np.exp(e_pow)
            return H
        else:
            self.factor = 1.e-2
            H_phi = self.far_field(x, y, z, t)
            H = np.array([0., 0., H_phi])
            H_x, H_y, H_z = spherical_to_cartesian(x, y, z, H)
            return [H_x, H_y, H_z]

    def S(self, x, y, z, t):
        E = self.E(x, y, z, t)
        H = self.H(x, y, z, t)
        if self.L == 0:
            self.factor = 0.5
        else:
            self.factor = 1.e-3
        return np.cross(E, H)

    def get_factor(self):
        return self.factor
