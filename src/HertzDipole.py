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
  r_cross_p = np.cross(r, p)
  rcrossp_cross_r = np.cross(r_cross_p, r)
  r_dot_p = np.dot(3 * r, p)
  r_dot_rdotp = np.dot(r, r_dot_p) - p

  c1 = (1 / (4 * np.pi * epsilon0))
  c2 = (omega**2 / (c**2 * r_norm))
  c3 = 1 / r_norm ** 3
  c4 = ((1j * omega) / (c * r_norm**2))
  c5 = (1j * omega * r_norm) / c
  c6 = -1j * omega * t

  E = c1 * (c2 * rcrossp_cross_r + ((c3 - c4) * r_dot_rdotp)) * np.exp(c5) * np.exp(c6)
  E = np.sum(np.real(E)**2)

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
  norm_p = np.sqrt(12 * pi * c * power / (mu0 * (2 * pi * frequency) ** 4))
  p = np.array([0, 0, norm_p])
  omega = 2 * np.pi * frequency

  nt = 100
  t0 = 1 / frequency / 10
  t1 = 5 / frequency
  nt = int(t1 / t0)
  t = np.linspace(t0, t1, nt)

  S = np.zeros((len(x), len(z)))

  for i in range(len(x)):
    for k in range(len(z)):
      S[i][k] = hertz_dipole(x[i], y[0], z[k], p, omega, 1E-10)

  plt.pcolor(x, z, S.T, cmap='hot')
  plt.xlabel(r'$x/$m')
  plt.ylabel(r'$z/$m')
  plt.savefig('img/dipole.png')
  plt.show()
  sys.exit(0)
