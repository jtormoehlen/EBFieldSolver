import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

epsilon_0 = 8.85E-12
mu_0 = 4. * np.pi * 10.0E-7
c = 299792458.
Z_0 = np.sqrt(2 * mu_0) / epsilon_0


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
        return self.epsilon * self.q * ((r - self.r0) / r_r0_norm ** 3)

    def phi(self, x, y, z, t=0):
        r = np.array([x, y, z])
        r_r0_norm = np.linalg.norm(r - self.r0)
        return self.epsilon * self.q * (1 / r_r0_norm) * np.array([1, 0, 0])


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
        self.P = power
        self.T = 1 / frequency
        self.omega = 2 * np.pi * frequency
        self.lambda_0 = c / frequency
        self.k_0 = (2 * np.pi) / self.lambda_0
        self.L = l * self.lambda_0
        self.h = self.L / 2

    def p(self, d):
        e_z = np.array([0, 0, 1])
        I_0 = np.sqrt((48 * np.pi * self.P) / (Z_0 * self.k_0 ** 2 * d ** 2))
        return (1j * I_0 * d) / (2 * self.omega) * e_z

    def E(self, x, y, z, t=0):
        if self.L == 0:
            p = self.p(self.lambda_0 / 10)
            r_dir = np.array([x, y, z]) - self.r0
            r = np.linalg.norm(r_dir)
            n = r_dir / r
            n_cross_p = np.cross(n, p)
            ncrossp_cross_n = np.cross(n_cross_p, n)
            n_dot_p = np.dot(n, p)
            n_dot_ndotp = np.dot(3 * n, n_dot_p) - p
            const = 1 / (4 * np.pi * epsilon_0)
            exp_rt = np.exp(1j * (self.k_0 * r - self.omega * t))
            return const * (self.k_0 ** 2 * ncrossp_cross_n * (exp_rt / r) +
                            n_dot_ndotp * ((1 / r ** 3) - ((1j * self.k_0) / r ** 2)) * exp_rt)
        else:
            return self.A(x, y, z, t) * (1j * Z_0) / (mu_0 * self.k_0)

    def H(self, x, y, z, t=0):
        if self.L == 0:
            p = self.p(self.lambda_0 / 10)
            r_dir = np.array([x, y, z]) - self.r0
            r = np.linalg.norm(r_dir)
            n = r_dir / r
            n_cross_p = np.cross(n, p)
            const = (c * self.k_0 ** 2) / (4 * np.pi)
            exp_rt = np.exp(1j * (self.k_0 * r - self.omega * t))
            return const * n_cross_p * (exp_rt / r) * (1 - (1 / (1j * self.k_0 * r)))
        else:
            return self.A(x, y, z, t) * (1 / mu_0)

    def A(self, x, y, z, t=0):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        p = self.p(self.L)
        if self.L == 0:
            const = -(1j * mu_0 * self.omega) / (4 * np.pi)
            return const * p * ((np.exp(1j * (self.k_0 * r - self.omega * t))) / r)
        else:
            theta = np.arccos(z / r)
            f_theta_phi = (np.cos(self.k_0 * self.h * np.cos(theta)) - np.cos(self.k_0 * self.h)) / np.sin(theta) ** 2
            exp_t = np.exp(1j * (self.k_0 * r - self.omega * t))
            I_0 = np.sqrt((48 * np.pi * self.P) / (Z_0 * self.k_0 ** 2 * self.L ** 2))
            A_z = ((mu_0 * I_0 * exp_t) / (2 * np.pi * self.k_0 * r)) * f_theta_phi
            return np.array([0, 0, A_z])
