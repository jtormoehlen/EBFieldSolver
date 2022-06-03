import numpy as np
import matplotlib.pyplot as plt
from lib.FieldCalculator import field, rot, grad, mesh
from lib.FieldPlot import N, N3D
from lib.FieldObject import Charge, Current

plt.style.use('./lib/figstyle.mpstyle')


def draw_loop(plot3d, fobs, ax):
    if plot3d:
        fob = fobs[0]
        r = fob.r0
        for i in range(len(r)):
            for j in range(0, 3, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]],
                             [r[i][1], r[i + 1][1]],
                             [r[i][2], r[i + 1][2]],
                             color='black', ls='-', lw=2.)
        ax.plot([r[len(r) - 1][0], r[0][0]],
                [r[len(r) - 1][1], r[0][1]],
                [r[len(r) - 1][2], r[0][2]],
                color='black', ls='-', lw=2.)

    else:
        fob = fobs[0]
        r = fob.r0
        for i in range(len(r)):
            for j in range(0, 2, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]], [r[i][1], r[i + 1][1]],
                             color='black', ls='-', lw=5.0)
        ax.plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                color='black', ls='-', lw=5.0)


def draw_charges(plot3d, fobs, ax):
    if plot3d:
        R = 0.5
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            ax.plot_surface(x + 2 * fob.r0[0], y + 2 * fob.r0[1], z + 2 * fob.r0[2], color=color)
    else:
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            circle = plt.Circle((fob.r0[0], fob.r0[1]), 0.25, color=color)
            ax.add_patch(circle)


def anim():
    fig = plt.figure()
    # for i in np.arange(1, 5, 1):
    #     ax = fig.add_subplot(2, 2, i)
    #     ax.set_xlabel('$x$', labelpad=0)
    #     ax.set_ylabel('$z$', labelpad=0)
    #     ax.tick_params(direction='in', right=True, top=True, which='minor')
    #     c = 'a'
    #     ax.set_title(f'({chr(ord(c[0]) + (i - 1))})')
    #     xy.extend([min(xy), max(xy)])
    #     x, y, z = mesh(xy, N)
    #     n = round(N / 2)
    #     ax.set_xlim(np.min(x), np.max(x))
    #     ax.set_ylim(np.min(y), np.max(y))
    #     x = x[:, n, :]
    #     y = z[:, n, :]
    #     f_x, f_y, f_c = dynamic(xy, 0, fobs[i - 1], ffunc, 1.)
    #     fmin = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    #     f_x, f_y, f_c = dynamic(xy, 0, fobs[i - 1], ffunc, fmin)
    #     ax.quiver(x, y, f_x, f_y, f_c, cmap='coolwarm', pivot='mid')

    # x, y, z = mesh(xy, N)
    # n = round(N / 2)
    #
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.tick_params(direction='in', right=True, top=True, which='minor')
    # ax1.set_xlabel('$x$', labelpad=0)
    # ax1.set_ylabel('$z$', labelpad=0)
    # ax1.set_title('(a)')
    # ax1.tick_params(direction='in', right=True, top=True, which='minor')
    # f_x, f_y, f_c = dynamic(xy, 0, fobs, 'E', 1.)
    # fmin = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    # f_x, f_y, f_c = dynamic(xy, 0, fobs, 'E', fmin)
    # ax1.quiver(x[:, n, :], z[:, n, :], f_x, f_y, f_c, cmap='coolwarm', pivot='mid')
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    #
    # ax2.tick_params(direction='in', right=True, top=True, which='minor')
    # ax2.set_xlabel('$x$', labelpad=0)
    # ax2.set_ylabel('$y$', labelpad=0)
    # ax2.set_title('(b)')
    # f_x, f_y, f_c = dynamic(xy, 0, fobs, 'H', 1.)
    # fmin = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    # f_x, f_y, f_c = dynamic(xy, 0, fobs, 'H', fmin)
    # ax2.quiver(x[:, :, n], y[:, :, n], f_x, f_y, f_c, cmap='cool', pivot='mid')
    # plt.show()


def main():
    """example: electrical quadrupole"""
    q = x = y = 1  # charge amount and position comps
    charges = [Charge(-q, -x, y),  # -q at (-1,1)
               Charge(q, x, y),  # q at (1,1)
               Charge(q, -x, -y),  # q at (-1,-1)
               Charge(-q, x, -y)]  # -q at (1,-1)

    """example: elliptical conductor loop"""
    I = 1.  # current
    r = []  # current elements r_i=(x_i,y_i,z_i)
    dr = []  # current element direction dr_i=(dr_x_i,dr_y_i,dr_z_i)
    a = 0.5  # semi-minor axis
    b = 1.5  # semi-major axis
    N = 20  # approximation order
    phi_n = np.linspace(0., 2 * np.pi, N, endpoint=False)  # discrete angle array
    dphi = 2 * np.pi / N  # length of dr
    for phi in phi_n:  # for each discrete angle
        r_x = dr_x = 0.  # loop in (y, z)-plane
        r_y = a * np.cos(phi)
        r_z = b * np.sin(phi)
        dr_y = -a * np.sin(phi) * dphi
        dr_z = b * np.cos(phi) * dphi
        r.append(np.array([r_x, r_y, r_z]))
        dr.append(np.array([dr_x, dr_y, dr_z]))
    currents = [Current(I, r, dr)]

    xyz = [-2., 2., -2., 2., -2., 2.]
    n = round(N / 2)
    x, y, z = mesh(xyz, N, rc='xy')
    A_x, A_y, A_z = field(xyz, N, currents, ffunc='A', rc='xy')
    B_y, B_x, B_z = -rot(A_y, A_x, A_z)
    # phi, phi_y, phi_z = field(xyz, N, charges, ffunc='phi', rc='xy')
    # E_y, E_x, E_z = -grad(phi)
    f_1, f_2, f_3 = field(xyz, N3D, currents, ffunc='B')
    x_1, x_2, x_3 = mesh(xyz, N3D)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('(a)')
    lev = np.linspace(np.min(A_z[:, :, n]) / 10, np.max(A_z[:, :, n]) / 10, 4)
    ax1.contour(x[:, :, n], y[:, :, n], A_z[:, :, n],
                lev, colors='k', alpha=0.5)
    f_norm = np.hypot(B_x[:, :, n], B_y[:, :, n])
    ax1.streamplot(x[:, :, n], y[:, :, n], B_x[:, :, n], B_y[:, :, n],
                   color=np.log(f_norm), cmap='binary', zorder=0, density=1, arrowsize=0.5)
    # draw_charges(False, charges, ax1)
    draw_loop(False, currents, ax1)
    ax1.minorticks_on()
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y), np.max(y))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.tick_params(direction='in', right=True, top=True, which='minor')

    # # c = np.log(np.hypot(E_1, E_2, E_3))
    # # c = (c.ravel() - c.min()) / c.ptp()
    # # c = np.concatenate((c, np.repeat(c, 2)))
    # # c = plt.cm.binary(c)

    p0 = np.zeros_like(x_3)
    for i in range(len(p0)):
        for j in range(len(p0)):
            p0[i, -1, j] = 1.
            p0[0, i, j] = 1.
            p0[i, j, 0] = 1.

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('(b)')
    # ax2.view_init(22.5, -67.5)
    ax2.view_init(22.5, -45)
    # draw_charges(True, charges, ax2)
    draw_loop(True, currents, ax2)
    ax2.quiver(x_1, x_2, x_3, f_1 * p0, f_2 * p0, f_3 * p0, colors='black',
               arrow_length_ratio=0.5, pivot='middle',
               length=max(xyz) / N3D, normalize=True,
               alpha=0.5)
    ax2.set_xlim3d(np.min(x_1), np.max(x_1))
    ax2.set_ylim3d(np.min(x_2), np.max(x_2))
    ax2.set_zlim3d(np.min(x_3), np.max(x_3))
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$z$')

    plt.show()


if __name__ == "__main__":
    main()
