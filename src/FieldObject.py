import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

epsilon_0 = 8.85E-12
mu_0 = 4 * np.pi * 10E-7
c = 299792458.


class Charge:
    const = 1 / (4 * np.pi * epsilon_0)

    def __init__(self, q, r0):
        self.q = q
        self.r0 = r0
        self.R = 0.25

    def form(self):
        if self.q == 0:
            q_color = 'black'
        elif self.q < 0:
            q_color = 'blue'
        else:
            q_color = 'red'
        circle = plt.Circle(self.r0, self.R, color=q_color)
        plt.gca().add_patch(circle)

    def E_field(self, x, y, z):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        Ex = self.const * self.q * (x - self.r0[0]) / r ** 3
        Ey = self.const * self.q * (y - self.r0[1]) / r ** 3
        Ez = 0.
        return [Ex, Ey, Ez]

    def phi_field(self, x, y, z):
        r = np.hypot(x - self.r0[0], y - self.r0[1])
        phi = self.const * self.q * (1 / r)
        return [phi, 0., 0.]


class Conductor:
    const = (mu_0 / (2 * np.pi))

    def __init__(self, I, r0):
        self.I = I
        self.R = 0.5
        self.r0 = r0

    def form(self):
        if self.I == 0:
            I_direction = '*'
            I_color = 'black'
            size = self.R + 100
        elif self.I > 0:
            I_direction = 'o'
            I_color = 'black'
            size = self.R + 10
        else:
            I_direction = 'x'
            I_color = 'black'
            size = self.R + 100
        circle = plt.Circle(self.r0, self.R, edgecolor='black', facecolor='white')
        plt.gca().add_patch(circle)
        plt.scatter(self.r0[0], self.r0[1], size, I_color, I_direction, zorder=2)

    def B_field(self, x, y, z):
        mag = self.const * (self.I / np.hypot(x - self.r0[0], y - self.r0[1]))
        Bx = mag * (np.cos(np.arctan2(x - self.r0[0], y - self.r0[1])))
        By = mag * (-np.sin(np.arctan2(x - self.r0[0], y - self.r0[1])))
        Bz = 0.
        return [Bx, By, Bz]

    def A_field(self, x, y, z):
        mag = -(1. / 2.) * self.const * self.I
        Ax = 0.
        Ay = 0.
        Az = mag * np.log(np.sqrt((x - self.r0[0])**2 + (y - self.r0[1])**2))
        return [Ax, Ay, Az]


class HertzDipole:
    def __init__(self, r0, frequency, power):
        self.r0 = r0
        self.R = 0.05
        self.frequency = frequency
        self.T = 1. / frequency
        self.power = power
        self.omega = 2. * np.pi * self.frequency
        self.wavelength = c / self.frequency
        self.k = (2. * np.pi) / self.wavelength
        self.p_z = np.sqrt(12. * np.pi * c * self.power / (mu_0 * self.omega ** 4))

    def form(self):
        ellipse = patch.Ellipse(self.r0, 0.1, 0.2, color='grey', alpha=0.5)
        plt.gca().add_patch(ellipse)

    def p(self, t):
        e_z = np.array([0., 0., 1.])
        p = self.p_z * np.exp(-1j * self.omega * t) * e_z
        return p

    def E_field(self, x, y, z, t):
        p = self.p(t)
        r = np.array([x, y, z]) - self.r0
        r_norm = np.linalg.norm(r)
        r = r / r_norm
        r_cross_p = np.cross(r, p)
        rcrossp_cross_r = np.cross(r_cross_p, r)
        r_dot_p = np.dot(r, p)
        r_dot_rdotp = np.dot(3 * r, r_dot_p) - p

        c1 = (self.omega ** 3 / (4. * np.pi * epsilon_0 * c ** 3))
        c2 = (self.omega * r_norm) / c
        c3 = 1. / c2
        c4 = 1. / c2 ** 3
        c5 = 1.j / c2 ** 2
        c6 = 1.j * (c2 - (self.omega * t))

        E = c1 * ((rcrossp_cross_r * c3) + (r_dot_rdotp * (c4 - c5))) * np.exp(c6)
        return E

    def H_field(self, x, y, z, t):
        p = self.p(t)
        r = np.array([x, y, z]) - self.r0
        r_norm = np.linalg.norm(r)
        r = r / r_norm
        r_cross_p = np.cross(r, p)

        c1 = (self.omega ** 3 / (4. * np.pi * c ** 2))
        c2 = (self.omega * r_norm) / c
        c3 = 1. / c2
        c4 = 1.j / c2 ** 2
        c5 = 1.j * (c2 - (self.omega * t))

        H = c1 * r_cross_p * (c3 + c4) * np.exp(c5)
        return H

    def S_field(self, x, y, z, t):
        E = self.E_field(x, y, z, t)
        H = self.H_field(x, y, z, t)
        return np.cross(E, H)
