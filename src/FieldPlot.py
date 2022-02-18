import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from FieldCalculator import rot, grad, field, field_round, mesh, phi_unit

N_XYZ = 30
N_XYZ_3D = 10


def static(xy, fobs, ffunc, nabla='', view='xy'):
    """
    Build stationary field.
    :param view: view plane
    :param xy: spatial coords (x, y) both from -xy_max to xy_max
    :param fobs: list contains all field emitting objects
    :param ffunc: desired field to compute
    :param nabla: pass field operator (rot or grad); default: none
    """
    f_x, f_y, f_z = field(xy, N_XYZ, fobs, ffunc=ffunc, index='xy')
    if nabla == 'rot':
        potential_lines(xy, f_z)
        f_y, f_x, f_z = -rot(f_y, f_x, f_z)
    if nabla == 'grad':
        potential_lines(xy, f_x)
        f_y, f_x, f_z = -grad(f_x)
    field_lines(xy, f_x, f_y, f_z, ffunc, fobs, view)


def static3d(xyz, fobs, ffunc, nabla=''):
    """
    Build stationary 3d-field.
    :param xyz: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param fobs: list contains all field emitting objects
    :param ffunc: desired field to compute
    :param nabla: pass field operator (rot or grad); default: none
    """
    f_x, f_y, f_z = field(xyz, N_XYZ_3D, fobs, ffunc=ffunc)
    if nabla == 'rot':
        f_x, f_y, f_z = rot(f_x, f_y, f_z)
    elif nabla == 'grad':
        f_x, f_y, f_z = -grad(f_x)
    arrow_field3d(xyz, f_x, f_y, f_z, ffunc, fobs)


def dynamic(xy, t, fobs, ffunc, f_xy_min=1):
    """
    Build dynamic field.
    :param xy: spatial coords (x, y) both from -xy_max to xy_max
    :param t: time coord
    :param fobs: list contains all field emitting objects
    :param ffunc: desired field to compute
    :param f_xy_min: minimum-length of a vector in field
    :return: E- or H-field at time t
    """
    n = round(N_XYZ / 2)
    f_x, f_y, f_z = field(xy, N_XYZ, fobs, t=t, ffunc=ffunc)
    if ffunc == 'E':
        if fobs[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        field_round(f_x[:, n, :], f_z[:, n, :], f_xy_min, fobs[0])
        f_xz_norm = np.sqrt(f_x[:, n, :] ** 2 + f_z[:, n, :] ** 2)
        return f_x[:, n, :], f_z[:, n, :], f_z[:, n, :] / f_xz_norm
    elif ffunc == 'H':
        if fobs[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        field_round(f_x[:, :, n], f_y[:, :, n], f_xy_min, fobs[0])
        f_xy_norm = np.sqrt(f_x[:, :, n] ** 2 + f_y[:, :, n] ** 2)
        return f_x[:, :, n], f_y[:, :, n], (f_x[:, :, n] / f_xy_norm) * phi_unit(xy, N_XYZ)


def arrow_field3d(xyz, f_x, f_y, f_z, ffunc, fobs=(None,)):
    """
    Plot 3d-field.
    :param xyz: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param f_x: 3d-grid of x-coord
    :param f_y: 3d-grid of y-coord
    :param f_z: 3d-grid of z-coord
    :param ffunc: desired field to draw
    :param fobs: desired field objects to draw
    """
    x, y, z = mesh(xyz, N_XYZ_3D)
    ax = plt.axes(projection='3d', proj_type='ortho')  # orthogonal projection: proj_type='ortho'
    # initial view angle: ax.view_init(theta, phi)
    ax.view_init(90, 90)
    ax.set_zlabel(r'$z$')
    n = round(N_XYZ / 2)
    ax.quiver(x, y, z, f_x[:, :, n], f_y[:, :, n], f_z, color='black',
              arrow_length_ratio=0.5, pivot='middle',
              length=max(xyz) / N_XYZ_3D, normalize=True,
              alpha=0.5)
    draw_objects(ffunc, fobs, plot3d=True)


def potential_lines(xy, f_xy, ffunc='', fobs=(None,)):
    """
    Plot contour lines depending on a height map.
    :param xy: spatial coords (x, y) both from -xy_max to xy_max
    :param f_xy: height map
    :param ffunc: desired field to draw
    :param fobs: desired field objects to draw
    """
    x, y, z = mesh(xy, N_XYZ, index='xy')
    n = round(N_XYZ / 2)
    f_xy_levels = np.linspace(np.min(f_xy[:, :, n]) / 10, np.max(f_xy[:, :, n]) / 10, 4)
    plt.contour(x[:, :, n], y[:, :, n], f_xy[:, :, n],
                f_xy_levels, colors='k', alpha=0.5)
    draw_objects(ffunc, fobs)


def field_lines(xy, f_x, f_y, f_z, ffunc, fobs=(None,), view='xy'):
    """
    Plot field lines.
    :param xy: spatial coords (x, y) both from -xy_max to xy_max
    :param f_x: 3d-grid of x-coord
    :param f_y: 3d-grid of y-coord
    :param f_z: 3d-grid of z-coord
    :param ffunc: desired field to draw
    :param fobs: desired field objects to draw
    :param view: view plane
    """
    x, y, z = mesh(xy, N_XYZ, index='xy')
    n = round(N_XYZ / 2)
    if view == 'xz':
        f_1 = f_x[:, n, :]
        f_2 = f_z[:, n, :]
    elif view == 'yz':
        f_1 = f_y[n, :, :]
        f_2 = f_z[n, :, :]
    else:
        f_1 = f_x[:, :, n]
        f_2 = f_y[:, :, n]
    f_norm = np.hypot(f_1, f_2)
    plt.streamplot(x[:, :, n], y[:, :, n], f_1, f_2,
                   color=np.log(f_norm), cmap='binary', zorder=0, density=2)
    draw_objects(ffunc, fobs)


def draw_objects(ffunc, fobs, plot3d=False):
    """
    Helper-function: Decide whether charge(s) or current loop(s) should be drawn in 2d/3d
    :param ffunc: corresponding field
    :param fobs: desired field objects to draw
    :param plot3d: true if 3d, false else
    """
    if not fobs[0] is None:
        if ffunc == 'E' or ffunc == 'phi':
            draw_charges(plot3d, fobs)
        elif ffunc == 'B' or ffunc == 'A':
            draw_loop(plot3d, fobs)


def draw_loop(plot3d=False, fobs=(None,), closed=True):
    """
    Draw current loop in 2d or 3d.
    :param plot3d: true if 3d, false else
    :param fobs: desired field objects to draw
    :param closed: true if loop is closed, false else
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
        if closed:
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
        if closed:
            plt.plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                     color='black', ls='-', lw=5.0, alpha=0.5)


def draw_charges(plot3d=False, fobs=(None,)):
    """
    Draw discrete charge distribution in 2d or 3d.
    :param plot3d: true if 3d, false else
    :param fobs: desired field objects to draw
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
