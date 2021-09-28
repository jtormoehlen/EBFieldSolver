import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

epsilon_0 = 8.85e-12
mu_0 = 4. * np.pi * 10.0e-7
c = 299792458.
Z_0 = np.sqrt(mu_0 / epsilon_0)


class Charge:
    epsilon = 1 / (4 * np.pi * epsilon_0)

    def __init__(self, q, r0_x, r0_y, r0_z=0):
        self.q = q
        self.r0 = np.array([r0_x, r0_y, r0_z])
        self.R = 0.25

    def form(self):
        qcolor = 'blue' if self.q < 0 else 'red'
        circle = plt.Circle((self.r0[0], self.r0[1]), self.R, color=qcolor)
        plt.gca().add_patch(circle)

    def E(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        E = self.epsilon * self.q * ((r - self.r0) / r_r0_norm ** 3)
        return E

    def phi(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        phi = self.epsilon * self.q * (1 / r_r0_norm) * np.array([1, 0, 0])
        return phi


class Current:
    mu = (mu_0 / (4. * np.pi))

    def __init__(self, I, r0, dl):
        self.I = I
        self.r0 = r0
        self.dl = dl

    def form(self):
        y = 0
        # isymbol = 'x' if self.v[2] < 0 else 'o'
        # circle = plt.Circle((self.r0[0], self.r0[2]), self.R, color='grey')
        # plt.gca().add_patch(circle)
        # # plt.scatter(self.r0[0], self.r0[2], self.R*10, 'black', isymbol, zorder=2)

    def B(self, x, y, z, t=0):
        r = np.array([x, y, z])
        B = 0
        for i in range(len(self.r0)):
            r_r0_norm = np.linalg.norm(r - self.r0[i])
            dl_cross_r_r0 = np.cross(self.dl[i], r - self.r0[i])
            B += self.mu * ((self.I * dl_cross_r_r0) / (r_r0_norm ** 3))
        return B

    def A(self, x, y, z, t=0):
        r = np.array([x, y, z])
        A = 0
        for i in range(len(self.r0)):
            r_r0_norm = np.linalg.norm(r - self.r0[i])
            A += self.mu * ((self.I * self.dl[i]) / r_r0_norm)
        return A


class Antenna:
    def __init__(self, frequency, power, l=0, x=0, y=0, z=0):
        self.r0 = np.array([x, y, z])
        self.T = 1 / frequency
        self.omega = 2 * np.pi * frequency
        self.lambda_0 = c / frequency
        self.k_0 = (2 * np.pi) / self.lambda_0
        self.L = l * self.lambda_0
        self.h = self.L / 2
        self.I_0 = power / 1e3
        self.p_z = np.sqrt(12 * np.pi * c * power / (mu_0 * self.omega ** 4))

    def p(self, t):
        e_z = np.array([0, 0, 1])
        return self.p_z * np.exp(-1j * self.omega * t) * e_z

    def E(self, x, y, z, t=0):
        if self.L == 0:
            p = self.p(t)
            r = np.array([x, y, z]) - self.r0
            r_norm = np.linalg.norm(r)
            r = r / r_norm
            r_cross_p = np.cross(r, p)
            rcrossp_cross_r = np.cross(r_cross_p, r)
            r_dot_p = np.dot(r, p)
            r_dot_rdotp = np.dot(3 * r, r_dot_p) - p

            const = (self.omega ** 3 / (4 * np.pi * epsilon_0 * c ** 3))
            rho = (self.omega * r_norm) / c
            far = 1 / rho
            near_3 = 1 / rho ** 3
            near_2 = 1j / rho ** 2
            e_pow = 1j * (rho - (self.omega * t))
            return const * ((rcrossp_cross_r * far) + (r_dot_rdotp * (near_3 - near_2))) * np.exp(e_pow)
        else:
            A = self.A(x, y, z, t)
            return A * (1j * Z_0) / (mu_0 * self.k_0)

    def H(self, x, y, z, t=0):
        if self.L == 0:
            p = self.p(t)
            r = np.array([x, y, z]) - self.r0
            r_norm = np.linalg.norm(r)
            r = r / r_norm
            r_cross_p = np.cross(r, p)

            const = (self.omega ** 3 / (4 * np.pi * c ** 2))
            rho = (self.omega * r_norm) / c
            far = 1 / rho
            near_2 = 1j / rho ** 2
            e_pow = 1j * (rho - (self.omega * t))
            return const * r_cross_p * (far + near_2) * np.exp(e_pow)
        else:
            return self.A(x, y, z, t) * (1 / mu_0)

    def A(self, x, y, z, t=0):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        f_theta_phi = (np.cos(self.k_0 * self.h * np.cos(theta)) - np.cos(self.k_0 * self.h)) / np.sin(theta) ** 2
        e_t = np.exp(1j * (self.k_0 * r - self.omega * t))
        I = self.I_0 / np.sin(self.k_0 * self.h)
        A_z = ((mu_0 * I * e_t) / (2 * np.pi * self.k_0 * r)) * f_theta_phi
        return np.array([0, 0, A_z])

    def S(self, x, y, z, t):
        return np.cross(self.E(x, y, z, t), self.H(x, y, z, t))
