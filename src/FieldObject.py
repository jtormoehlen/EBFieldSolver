import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import FieldOperator as fo

epsilon_0 = 8.85e-12
mu_0 = 4. * np.pi * 10.e-7
c = 299792458.
Z_0 = np.sqrt(mu_0 / epsilon_0)


class Charge:
    epsilon = 1. / (4. * np.pi * epsilon_0)
    mu = (mu_0 / (4. * np.pi))

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

    def E(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        E = self.epsilon * self.q * ((r - self.r0) / r_r0_norm ** 3)
        return E

    def phi(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        phi = self.epsilon * self.q * (1 / r_r0_norm) * np.array([1., 0., 0.])
        return phi

    def B(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        v_cross_r_r0 = np.cross(self.v, r - self.r0)
        B = self.mu * ((self.q * v_cross_r_r0) / (r_r0_norm ** 3))
        return B

    def A(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        A = self.mu * ((self.q * self.v) / r_r0_norm)
        return A


class Antenna:
    def __init__(self, frequency, power, l=0., x=0., y=0., z=0.):
        self.r0 = np.array([x, y, z])
        self.T = 1. / frequency
        self.omega = 2. * np.pi * frequency
        self.lambda_0 = c / frequency
        self.k_0 = (2. * np.pi) / self.lambda_0
        self.L = l * self.lambda_0
        self.h = self.L / 2.
        self.I_0 = power / 1.e3
        self.p_z = np.sqrt(12. * np.pi * c * power / (mu_0 * self.omega ** 4))
        self.factor = 1.

    def p(self, t):
        e_z = np.array([0., 0., 1.])
        p = self.p_z * np.exp(-1j * self.omega * t) * e_z
        return p

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
            return const * ((rcrossp_cross_r * far) + (r_dot_rdotp * (near_3 - near_2))) * np.exp(e_pow)
        else:
            self.factor = .1
            return (1.j * Z_0) / (mu_0 * self.k_0)

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
            return const * r_cross_p * (far + near_2) * np.exp(e_pow)
        else:
            self.factor = .005
            return 1. / mu_0

    def A(self, x, y, z, t, nabla=''):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        f_theta_phi = (np.cos(self.k_0 * self.h * np.cos(theta)) - np.cos(self.k_0 * self.h)) / np.sin(theta) ** 2
        e_t = np.exp(1.j * (self.k_0 * r - self.omega * t))
        I = self.I_0 / np.sin(self.k_0 * self.h)
        A_z = ((mu_0 * I * e_t) / (2 * np.pi * self.k_0 * r)) * f_theta_phi
        if nabla == 'rot':
            return np.array([0., 0., A_z * self.H(x, y, z, t)])
        elif nabla == 'rotrot':
            return np.array([0., 0., A_z * self.E(x, y, z, t)])
        else:
            return np.array([0., 0., A_z])

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
