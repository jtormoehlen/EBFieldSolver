import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from FieldCalculator import rot, grad, field, field_round, mesh, phi_unit

N_XYZ = 30
N_XYZ_3D = 10


def static(xy, fobs, ffunc, nabla):
    """
    Stationary 2d-field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param fobs: list of field objects
    :param ffunc: field function
    :param nabla: nabla operator {'' (none default), 'rot', 'grad'}
    """
    f_x, f_y, f_z = field(xy, N_XYZ, fobs, ffunc=ffunc, index='xy')
    if nabla == 'rot':
        potential_lines(xy, f_z, ffunc)
        f_y, f_x, f_z = -rot(f_y, f_x, f_z)
    elif nabla == 'grad':
        potential_lines(xy, f_x, ffunc)
        f_y, f_x, f_z = -grad(f_x)
    field_lines(xy, [f_x, f_y], ffunc, fobs)


def static3d(xyz, fobs, ffunc, nabla, view):
    """
    Stationary 3d-field.
    :param xyz: list of spatial coords [x1, x2, y1, y2, z1, z2]
    :param fobs: list of field objects
    :param ffunc: field function
    :param nabla: nabla operator {'' (none default), 'rot', 'grad'}
    :param view: view plane {''(none default), 'xy', 'xz', 'yz'}
    """
    f_x, f_y, f_z = field(xyz, N_XYZ_3D, fobs, ffunc=ffunc)
    if nabla == 'rot':
        f_x, f_y, f_z = rot(f_x, f_y, f_z)
    elif nabla == 'grad':
        f_x, f_y, f_z = -grad(f_x)
    arrow_field3d(xyz, [f_x, f_y, f_z], ffunc, view, fobs)


def dynamic(xy, t, fobs, ffunc, fmin):
    """
    Time-dependent field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param t: time coord >= 0
    :param fobs: list of field objects
    :param ffunc: field function
    :param fmin: minimal length in f
    :return: E- or H-field
    """
    n = round(N_XYZ / 2)
    f_x, f_y, f_z = field(xy, N_XYZ, fobs, t=t, ffunc=ffunc)
    if ffunc == 'E':
        if fobs[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        field_round(f_x[:, n, :], f_z[:, n, :], fmin, fobs[0])
        f_xz_norm = np.sqrt(f_x[:, n, :] ** 2 + f_z[:, n, :] ** 2)
        return f_x[:, n, :], f_z[:, n, :], f_z[:, n, :] / f_xz_norm
    elif ffunc == 'H':
        if fobs[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        field_round(f_x[:, :, n], f_y[:, :, n], fmin, fobs[0])
        f_xy_norm = np.sqrt(f_x[:, :, n] ** 2 + f_y[:, :, n] ** 2)
        return f_x[:, :, n], f_y[:, :, n], (f_x[:, :, n] / f_xy_norm) * phi_unit(xy, N_XYZ)


def arrow_field3d(xyz, f, ffunc, view, fobs=(None,)):
    """
    Plot arrows of 3d-field.
    :param xyz: list of spatial coords [x1, x2, y1, y2]
    :param f: field (f_x, f_y, f_z)
    :param ffunc: field function
    :param view: view plane {''(none default), 'xy', 'xz', 'yz'}
    :param fobs: list of field objects
    """
    f_x, f_y, f_z = f
    x, y, z = mesh(xyz, N_XYZ_3D)
    ax = plt.axes(projection='3d')  # orthogonal projection: proj_type='ortho'
    n = round(N_XYZ_3D / 2)
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_zlabel(r'$z$')
    p0 = np.zeros_like(z)
    for i in range(len(p0)):
        for j in range(len(p0)):
            if view == 'xy':
                p0[i, j, n] = 1.
                ax.view_init(90, -90)
            elif view == 'xz':
                p0[i, n, j] = 1.
                ax.view_init(0, -90)
            elif view == 'yz':
                p0[n, i, j] = 1.
                ax.view_init(0, 0)
            else:
                p0 = np.ones_like(z)
    ax.quiver(x, y, z, f_x * p0, f_y * p0, f_z * p0, color='black',
              arrow_length_ratio=0.5, pivot='middle',
              length=max(xyz) / N_XYZ_3D, normalize=True,
              alpha=0.5)
    draw_fobs(ffunc, fobs, plot3d=True)


def potential_lines(xy, f, ffunc, fobs=(None,)):
    """
    Plot contour lines of heights.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param f: heights
    :param ffunc: field function
    :param fobs: list of field objects
    """
    x, y, z = mesh(xy, N_XYZ, index='xy')
    n = round(N_XYZ / 2)
    f_xy_levels = np.linspace(np.min(f[:, :, n]) / 10, np.max(f[:, :, n]) / 10, 4)
    plt.contour(x[:, :, n], y[:, :, n], f[:, :, n],
                f_xy_levels, colors='k', alpha=0.5)
    draw_fobs(ffunc, fobs)


def field_lines(xy, f, ffunc, fobs=(None,)):
    """
    Plot lines of field.
    :param xy: list of spatial coords [x1, x2, y1, y2]
    :param f: field (f_x, f_y)
    :param ffunc: field function
    :param fobs: list of field objects
    """
    f_x, f_y = f
    x, y, z = mesh(xy, N_XYZ, index='xy')
    n = round(N_XYZ / 2)
    f_norm = np.hypot(f_x[:, :, n], f_y[:, :, n])
    plt.streamplot(x[:, :, n], y[:, :, n], f_x[:, :, n], f_y[:, :, n],
                   color=np.log(f_norm), cmap='binary', zorder=0, density=2)
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


def draw_loop(plot3d=False, fobs=(None,)):
    """
    Draw current loop.
    :param plot3d: plot 3d or 2d
    :param fobs: list of field objects
    """
    if plot3d:
        fob = fobs[0]
        r = fob.r0
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
        fob = fobs[0]
        r = fob.r0
        for i in range(len(r)):
            for j in range(0, 2, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]], [r[i][1], r[i + 1][1]],
                             color='black', ls='-', lw=5.0, alpha=0.5)
        plt.plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                 color='black', ls='-', lw=5.0, alpha=0.5)


def draw_charges(plot3d=False, fobs=(None,)):
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
