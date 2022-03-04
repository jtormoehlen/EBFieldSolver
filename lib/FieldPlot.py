import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from lib.FieldCalculator import rot, grad, field, field_limit, mesh, phi_unit

N = 30
N_3D = 10


def static(xy, fobs, ffunc, nabla):
    """
    Stationary 2d-field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param fobs: list of field objects
    :param ffunc: field function
    :param nabla: nabla operator {'' (none default), 'rot', 'grad'}
    """
    Fx, Fy, Fz = field(xy, N, fobs, ffunc=ffunc, rc='xy')
    if nabla == 'rot':
        pot_lines(xy, Fz, ffunc)
        Fy, Fx, Fz = -rot(Fy, Fx, Fz)
    elif nabla == 'grad':
        pot_lines(xy, Fx, ffunc)
        Fy, Fx, Fz = -grad(Fx)
    field_lines(xy, [Fx, Fy], ffunc, fobs)


def static3d(xyz, fobs, ffunc, nabla, view):
    """
    Stationary 3d-field.
    :param xyz: list of spatial coords [x1, x2, y1, y2, z1, z2]
    :param fobs: list of field objects
    :param ffunc: field function
    :param nabla: nabla operator {'' (none default), 'rot', 'grad'}
    :param view: view plane {''(none default), 'xy', 'xz', 'yz'}
    """
    Fx, Fy, Fz = field(xyz, N_3D, fobs, ffunc=ffunc)
    if nabla == 'rot':
        Fx, Fy, Fz = rot(Fx, Fy, Fz)
    elif nabla == 'grad':
        Fx, Fy, Fz = -grad(Fx)
    arrow_field3d(xyz, [Fx, Fy, Fz], ffunc, view, fobs)


def dynamic(xy, t, fobs, ffunc, fmin=1.):
    """
    Time-dependent field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param t: time coord >= 0
    :param fobs: list of field objects
    :param ffunc: field function
    :param fmin: minimal length in f
    :return: E- or H-field
    """
    n = round(N / 2)
    Fx, Fy, Fz = field(xy, N, fobs, t, ffunc)
    Fx, Fy, Fz = rot(Fx, Fy, Fz)
    if ffunc == 'E':
        Fx, Fy, Fz = rot(Fx, Fy, Fz)
        field_limit([Fx[:, n, :], Fz[:, n, :]], fmin)
        fnorm = np.hypot(Fx[:, n, :], Fz[:, n, :])
        return Fx[:, n, :], Fz[:, n, :], Fz[:, n, :] / fnorm
    elif ffunc == 'H':
        field_limit([Fx[:, :, n], Fy[:, :, n]], fmin)
        fnorm = np.hypot(Fx[:, :, n], Fy[:, :, n])
        return Fx[:, :, n], Fy[:, :, n], (Fx[:, :, n] / fnorm) * phi_unit(xy, N)


def arrow_field3d(xyz, F, ffunc, view, fobs=(None,)):
    """
    Plot arrows of 3d-field.
    :param xyz: list of spatial coords [x1, x2, y1, y2]
    :param F: field (Fx, Fy, Fz)
    :param ffunc: field function
    :param view: view plane {''(none default), 'xy', 'xz', 'yz'}
    :param fobs: list of field objects
    """
    Fx, Fy, Fz = F
    x, y, z = mesh(xyz, N_3D)
    ax = plt.axes(projection='3d')  # orthogonal projection: proj_type='ortho'
    n = round(N_3D / 2)
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_zlabel(r'$z$')
    p = np.zeros_like(z)
    for i in range(len(p)):
        for j in range(len(p)):
            if view == 'xy':
                p[i, j, n] = 1.
                ax.view_init(90, -90)
            elif view == 'xz':
                p[i, n, j] = 1.
                ax.view_init(0, -90)
            elif view == 'yz':
                p[n, i, j] = 1.
                ax.view_init(0, 0)
            else:
                p = np.ones_like(z)
    ax.quiver(x, y, z, Fx * p, Fy * p, Fz * p, color='black',
              arrow_length_ratio=0.5, pivot='middle',
              length=max(xyz) / N_3D, normalize=True,
              alpha=0.5)
    draw_fobs(ffunc, fobs, plot3d=True)


def pot_lines(xy, F, ffunc, fobs=(None,)):
    """
    Plot contour lines of heights.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param F: heights
    :param ffunc: field function
    :param fobs: list of field objects
    """
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N / 2)
    flvl = np.linspace(np.min(F[:, :, n]) / 10, np.max(F[:, :, n]) / 10, 4)
    plt.contour(x[:, :, n], y[:, :, n], F[:, :, n],
                flvl, colors='k', alpha=0.5)
    draw_fobs(ffunc, fobs)


def field_lines(xy, F, ffunc, fobs=(None,)):
    """
    Plot lines of field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param F: field (Fx, Fy)
    :param ffunc: field function
    :param fobs: list of field objects
    """
    Fx, Fy = F
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N / 2)
    fnorm = np.hypot(Fx[:, :, n], Fy[:, :, n])
    plt.streamplot(x[:, :, n], y[:, :, n], Fx[:, :, n], Fy[:, :, n],
                   color=np.log(fnorm), cmap='binary', zorder=0, density=2)
    draw_fobs(ffunc, fobs)


def draw_fobs(ffunc, fobs, plot3d=False):
    """
    Draw current loop or discrete charge distribution.
    :param ffunc: field function
    :param fobs: list of field objects
    :param plot3d: plot 3d or 2d
    """
    if not fobs[0] is None:
        if ffunc == 'E' or ffunc == 'phi':
            draw_charges(plot3d, fobs)
        elif ffunc == 'B' or ffunc == 'A':
            draw_loop(plot3d, fobs)


def draw_loop(plot3d, fobs):
    """
    Draw current loop.
    :param plot3d: plot 3d or 2d
    :param fobs: list of field objects
    """
    r = fobs[0].r0
    if plot3d:
        for i in range(len(r)):
            for j in range(0, 3, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]],
                             [r[i][1], r[i + 1][1]],
                             [r[i][2], r[i + 1][2]],
                             color='black', ls='-', lw=2., alpha=0.5)
        plt.plot([r[len(r) - 1][0], r[0][0]],
                 [r[len(r) - 1][1], r[0][1]],
                 [r[len(r) - 1][2], r[0][2]],
                 color='black', ls='-', lw=2., alpha=0.5)

    else:
        for i in range(len(r)):
            for j in range(0, 2, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]], [r[i][1], r[i + 1][1]],
                             color='black', ls='-', lw=5.0, alpha=0.5)
        plt.plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                 color='black', ls='-', lw=5.0, alpha=0.5)


def draw_charges(plot3d, fobs):
    """
    Draw discrete charge distribution.
    :param plot3d: plot 3d or 2d
    :param fobs: list of field objects
    """
    if plot3d:
        R = 0.5
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            plt.gca().plot_surface(x + 2 * fob.r0[0], y + 2 * fob.r0[1], z + 2 * fob.r0[2],
                                   color=color)
    else:
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            circle = plt.Circle((fob.r0[0], fob.r0[1]), 0.25, color=color)
            plt.gca().add_patch(circle)
