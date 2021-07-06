from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.integrate import tplquad
import sympy as sp

a = 1.0
Q = 1.0
R = 0.1
epsilon_0 = 8.85E-12


def charge_density_distribution(r):
    bounds = [0, R]
    if bounds[0] <= r <= bounds[1]:
        return a * r**2
    else:
        return 0


def electric_field(r):
    consts = (4 * np.pi) / epsilon_0
    A = 4 * np.pi * r ** 2
    f = lambda r: consts * charge_density_distribution(r) * r**2
    return quad(f, 0, r)[0] / A


def potential(r):
    c = 1 / (4 * np.pi * epsilon_0)
    g = lambda r: charge_density_distribution(r) * (1 / r)
    return quad(g, 0, r)[0] * c


# def sp_electric_field():
#     consts = 1 / (4 * np.pi * epsilon_0)
#     r = sp.Symbol('r')
#     u = sp.integrate(consts * charge_density_distribution(r) * r**2, r)
#     # print(u.evalf(subs={t: 1}))
#     print(u)
#     return u


r = np.linspace(0.001, 0.5, 200)
E = np.zeros(len(r))
phi = np.zeros(len(r))
for i in range(len(E)):
    E[i] = electric_field(r[i])
    # phi[i] = potential(r[i])
E_max = np.max(E)
# phi_max = np.max(phi)
plt.plot(r / R, E / E_max)
# plt.plot(r, phi / phi_max)
plt.xlabel(r"$r / R$")
plt.ylabel(r"$E / E_{max}$")
