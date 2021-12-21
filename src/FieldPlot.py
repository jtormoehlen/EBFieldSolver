import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from FieldCalculator import rot, grad, field, field_round, mesh, phi_unit

N_XYZ = 30
N_XYZ_3D = 10


def static(xy_max, objects, function, nabla=''):
    """
    Build stationary field.
    :param xy_max: spatial coords (x, y) both from -xy_max to xy_max
    :param objects: list contains all field emitting objects
    :param function: desired field to compute
    :param nabla: pass field operator (rot or grad); default: none
    """
    objects = o_to_olist(objects)
    f_x, f_y, f_z = field(xy_max, N_XYZ, objects, function=function, indexing='xy')
    if nabla == 'rot':
        potential_lines(xy_max, f_z)
        df_y, df_x, df_z = rot(f_y, f_x, f_z)
        field_lines(xy_max, -df_x, -df_y, function, objects)
    elif nabla == 'grad':
        potential_lines(xy_max, f_x)
        df_y, df_x, df_z = grad(f_x)
        field_lines(xy_max, -df_x, -df_y, function, objects)
    elif nabla == '':
        field_lines(xy_max, f_x, f_y, function, objects)


def static3d(xyz_max, objects, function, nabla=''):
    """
    Build stationary 3d-field.
    :param xyz_max: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param objects: list contains all field emitting objects
    :param function: desired field to compute
    :param nabla: pass field operator (rot or grad); default: none
    """
    objects = o_to_olist(objects)
    f_x, f_y, f_z = field(xyz_max, N_XYZ_3D, objects, function=function)
    if nabla == 'rot':
        f_x, f_y, f_z = rot(f_x, f_y, f_z)
        arrow_field3d(xyz_max, f_x, f_y, f_z, function, objects)
    if nabla == 'grad':
        f_x, f_y, f_z = grad(f_x)
        arrow_field3d(xyz_max, -f_x, -f_y, -f_z, function, objects)
    elif nabla == '':
        arrow_field3d(xyz_max, f_x, f_y, f_z, function, objects)


def dynamic(xy_max, t, objects, function, f_xy_min=1):
    """
    Build dynamic field.
    :param xy_max: spatial coords (x, y) both from -xy_max to xy_max
    :param t: time coord
    :param objects: list contains all field emitting objects
    :param function: desired field to compute
    :param f_xy_min: minimum-length of a vector in field
    :return: E- or H-field at time t
    """
    objects = o_to_olist(objects)
    f_x, f_y, f_z = field(xy_max, N_XYZ, objects, t=t, function=function)
    if function == 'E':
        if objects[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        plane = round(len(f_y) / 2)
        field_round(f_x[:, plane, :], f_z[:, plane, :], f_xy_min, objects[0])
        return f_x[:, plane, :], f_z[:, plane, :], f_z[:, plane, :]
    if function == 'H':
        if objects[0].L > 0:
            f_x, f_y, f_z = rot(f_x, f_y, f_z)
        plane = round(len(f_z) / 2)
        field_round(f_x[:, :, plane], f_y[:, :, plane], f_xy_min, objects[0])
        f_xy_norm = np.sqrt(f_x[:, :, plane] ** 2 + f_y[:, :, plane] ** 2)
        return f_x[:, :, plane], f_y[:, :, plane], (f_x[:, :, plane] / f_xy_norm) * phi_unit(xy_max, N_XYZ)


def arrow_field3d(xyz_max, f_x, f_y, f_z, function, field_objects=(None,)):
    """
    Plot 3d-field.
    :param xyz_max: spatial coords (x, y, z) all from -xyz_max to xyz_max
    :param f_x: 3d-grid of x-coord
    :param f_y: 3d-grid of y-coord
    :param f_z: 3d-grid of z-coord
    :param function: desired field to draw
    :param field_objects: desired field objects to draw
    """
    x, y, z = mesh(xyz_max, len(f_x))
    plt.subplot(projection='3d', label='none')
    if function == 'E' or function == 'phi':
        plt.gca().set_zlabel(r'$z/$d')
    elif function == 'B' or function == 'A':
        plt.gca().set_zlabel(r'$z/$R')
    # cmap = plt.get_cmap()
    # plane = round(len(f_z) / 2)
    # norm = np.hypot(f_x, f_y, f_z)
    plt.quiver(x, y, z, f_x, f_y, f_z, color='black',
               arrow_length_ratio=0.5, pivot='middle',
               length=xyz_max / len(f_x), normalize=True,
               alpha=0.5)
    draw_objects(function, field_objects, plot3d=True)


def potential_lines(xy_max, f_xy, function='', field_objects=(None,)):
    """
    Plot contour lines depending on a height map.
    :param xy_max: spatial coords (x, y) both from -xy_max to xy_max
    :param f_xy: height map
    :param function: desired field to draw
    :param field_objects: desired field objects to draw
    """
    x, y, z = mesh(xy_max, len(f_xy), indexing='xy')
    plane = round(len(z) / 2)
    f_xy_levels = np.linspace(np.min(f_xy[:, :, plane]) / 10, np.max(f_xy[:, :, plane]) / 10, 4)
    plt.contour(x[:, :, plane], y[:, :, plane], f_xy[:, :, plane], f_xy_levels, colors='k', alpha=0.5)
    draw_objects(function, field_objects)


def field_lines(xy_max, f_x, f_y, function, field_objects=(None,)):
    """
    Plot field lines.
    :param xy_max: spatial coords (x, y) both from -xy_max to xy_max
    :param f_x: 3d-grid of x-coord
    :param f_y: 3d-grid of y-coord
    :param function: desired field to draw
    :param field_objects: desired field objects to draw
    """
    x, y, z = mesh(xy_max, len(f_x), indexing='xy')
    plane = round(len(z) / 2)
    f_xy_norm = np.hypot(f_x[:, :, plane], f_y[:, :, plane])
    plt.streamplot(x[:, :, plane], y[:, :, plane], f_x[:, :, plane], f_y[:, :, plane],
                   color=np.log(f_xy_norm), cmap='binary', zorder=0, density=1)
    draw_objects(function, field_objects)


def draw_objects(function, field_objects, plot3d=False):
    """
    Helper-function: Decide whether charge(s) or current loop(s) should be drawn in 2d/3d
    :param function: corresponding field
    :param field_objects: desired field objects to draw
    :param plot3d: true if 3d, false else
    """
    if field_objects[0] is not None:
        if function == 'E' or function == 'phi':
            draw_charges(plot3d, field_objects)
        elif function == 'B' or function == 'A':
            draw_loop(plot3d, field_objects)


def o_to_olist(o):
    """
    Helper-function: put single object in a list
    :param o: single object
    :return: list containing single object
    """
    if not isinstance(o, list):
        olist = [o]
        return olist
    return o


def draw_loop(plot3d=False, objects=(None,), closed=True):
    """
    Draw current loop in 2d or 3d.
    :param plot3d: true if 3d, false else
    :param objects: desired field objects to draw
    :param closed: true if loop is closed, false else
    """
    if plot3d:
        o = objects[0]
        r = o.r0
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
        o = objects[0]
        r = o.r0
        for i in range(len(r)):
            for j in range(0, 2, 1):
                if i + 1 < len(r):
                    plt.plot([r[i][0], r[i + 1][0]], [r[i][1], r[i + 1][1]],
                             color='black', ls='-', lw=5.0, alpha=0.5)
        if closed:
            plt.plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                     color='black', ls='-', lw=5.0, alpha=0.5)


def draw_charges(plot3d=False, objects=(None,)):
    """
    Draw discrete charge distribution in 2d or 3d.
    :param plot3d: true if 3d, false else
    :param objects: desired field objects to draw
    """
    if plot3d:
        R = 0.5
        for o in objects:
            color = 'blue' if o.q < 0 else 'red'
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            plt.gca().plot_surface(x + 2 * o.r0[0], y + 2 * o.r0[1], z + 2 * o.r0[2],
                                   color=color)
    else:
        for o in objects:
            color = 'blue' if o.q < 0 else 'red'
            circle = plt.Circle((o.r0[0], o.r0[1]), 0.25, color=color)
            plt.gca().add_patch(circle)
