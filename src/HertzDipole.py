from __future__ import division

import sys

import numpy as np
from matplotlib import pyplot as plt

c = 299792458.
pi = np.pi
mu0 = 4 * pi * 1E-7
epsilon0 = 8.85 * 1E-12


def hertz_dipole(x, y, z, p, omega, t):
  r = np.array([x, y, z])
  r_norm = np.linalg.norm(r)
  r = r_norm * r
  r_cross_p = np.cross(r, p)
  rcrossp_cross_r = np.cross(r_cross_p, r)
  r_dot_p = np.dot(3 * r, p)
  r_dot_rdotp = np.dot(r, r_dot_p) - p

  c1 = (omega**3 / (4 * np.pi * epsilon0 * c**3))
  # c2 = (c / (omega * r_norm))
  c3 = (omega * r_norm) / c
  c4 = 1 / c3
  c5 = 1 / c3**3
  c6 = 1j / c3**2
  c7 = 1j * (c3 - (omega * t))

  E = c1 * ((rcrossp_cross_r * c4) + r_dot_rdotp * (c5 - c6)) * np.exp(c7)
  E = np.real(E)
  E = np.sqrt(E[0]**2 + E[2]**2)

  return E


if __name__ == "__main__":
  nx = 200
  x_max = 20.0
  x = np.linspace(-x_max, x_max, nx)

  y = np.array([0])

  nz = 100
  z_max = 10.0
  z = np.linspace(-z_max, z_max, nz)

  frequency = 1000E6
  power = 1
  # norm_p = np.sqrt(12 * pi * c * power / (mu0 * (2 * pi * frequency) ** 4))
  # print(norm_p)
  # p = np.array([0, 0, norm_p])

  omega = 2 * np.pi * frequency
  p_0 = np.array([0.0, 0.0, 1.0])
  p = p_0 * np.exp(-1j * omega * 0)

  nt = 100
  t0 = 1 / frequency / 10
  t1 = 5 / frequency
  nt = int(t1 / t0)
  t = np.linspace(t0, t1, nt)

  S = np.zeros((len(x), len(z)))

  for i in range(len(x)):
    for k in range(len(z)):
      S[i][k] = hertz_dipole(x[i], y[0], z[k], p, omega, 1)

  plt.pcolor(x, z, S.T, cmap='hot')
  plt.xlabel(r'$x/$m')
  plt.ylabel(r'$z/$m')
  plt.savefig('img/dipole1.png')
  plt.show()
  sys.exit(0)
