import numpy as np
import matplotlib.pyplot as plt
import FieldAnimation as fa
import FieldCalculator as fc
import FieldOperator as fo
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits import mplot3d
from matplotlib.patches import Circle, PathPatch, Ellipse


def static_field(xy_max, objects, function, nabla='', n_xy=30):
    if not isinstance(objects, list):
        objects = [objects]
    if nabla == 'rot':
        f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, function=function)
        df_x, df_y, df_z = fo.rot(f_x, f_y, f_z)
        z_plane = round(len(f_z) / 2)
        arrow_field(xy_max, df_x[:, :, z_plane], df_y[:, :, z_plane], normalize=True, show=True)
    if nabla == 'grad':
        f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, function=function)
        df_x, df_y, df_z = fo.grad(f_x)
        z_plane = round(len(f_z) / 2)
        arrow_field(xy_max, -df_x[:, :, z_plane], -df_y[:, :, z_plane], objects, normalize=True, show=True)
    if nabla == '':
        # f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, function=function, indexing='xy')
        # field_lines(xy_max, f_x, f_y, objects)
        f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, function='A', nabla='rotrot')
        plane = round(len(f_y) / 2)
        f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        x, y, z = fc.mesh3d(xy_max, n_xy)
        fc.field_round(f_x[:, plane, :], f_z[:, plane, :], xy_max, objects[0])
        plt.quiver(x[:, plane, :], z[:, plane, :],
                   f_x[:, plane, :], f_z[:, plane, :],
                   f_z[:, plane, :], cmap='cool')
        fa.render_frame()


def static_field3d(xyz_max, objects, function, nabla='', n_xyz=6):
    if not isinstance(objects, list):
        objects = [objects]
    if nabla == 'rot':
        f_x, f_y, f_z = fc.field(xyz_max, n_xyz, objects, function=function)
        f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        arrow_field3d(xyz_max, f_x, f_y, f_z, field_objects='loop')
    if nabla == 'grad':
        f_x, f_y, f_z = fc.field(xyz_max, n_xyz, objects, function=function)
        f_x, f_y, f_z = fo.grad(f_x)
        arrow_field3d(xyz_max, -f_x, -f_y, -f_z, field_objects='sphere')
    if nabla == '':
        f_x, f_y, f_z = fc.field(xyz_max, n_xyz, objects, function=function)
        arrow_field3d(xyz_max, f_x, f_y, f_z)


def init_dynamic_field(xy_max, n_xy, function):
    x, y, z = fc.mesh3d(xy_max, n_xy)
    if function == 'E':
        fa.render_frame(y_label='$z$', show=False)
        plane = round(len(y) / 2)
        return x[:, plane, :], z[:, plane, :]
    if function == 'H':
        fa.render_frame(show=False)
        plane = round(len(z) / 2)
        return x[:, :, plane], y[:, :, plane]


def dynamic_field(xy_max, t, objects, function, n_xy=30):
    if not isinstance(objects, list):
        objects = [objects]
    if function == 'E':
        if objects[0].L > 0:
            f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, t=t, function='A', nabla='rotrot')
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        else:
            f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, t=t, function=function)
        plane = round(len(f_y) / 2)
        fc.field_round(f_x[:, plane, :], f_z[:, plane, :], xy_max, objects[0])
        return f_x[:, plane, :], f_z[:, plane, :], f_z[:, plane, :]
    if function == 'H':
        if objects[0].L > 0:
            f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, t=t, function='A', nabla='rot')
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        else:
            f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, t=t, function=function)
        plane = round(len(f_z) / 2)
        fc.field_round(f_x[:, :, plane], f_y[:, :, plane], xy_max, objects[0])
        f_xy_norm = np.sqrt(f_x[:, :, plane] ** 2 + f_y[:, :, plane] ** 2)
        return f_x[:, :, plane], f_y[:, :, plane], (f_x[:, :, plane] / f_xy_norm) * fc.phi_unit(xy_max, n_xy)
    # S: f_x * fc.radius_unit(xy_max, n_xy)


def arrow_field(xy_max, f_x, f_y, field_objects=(None,), normalize=False, cfunc=None, show=True):
    x, y = fc.mesh(xy_max, len(f_x))
    if normalize:
        f_xy_norm = np.sqrt(f_x ** 2 + f_y ** 2)
        colorf = f_xy_norm
    else:
        f_xy_norm = 1.
        colorf = np.sqrt(f_x ** 2 + f_y ** 2)
    if cfunc is None:
        cfunc = np.log(colorf)
    else:
        cfunc = cfunc
    plt.quiver(x, y, f_x / f_xy_norm, f_y / f_xy_norm, cfunc, cmap='cool')
    # plt.quiver(x, y, f_x / f_xy_norm, f_y / f_xy_norm)
    # draw(field_objects)
    fa.render_frame() if show else 0


def arrow_field3d(xyz_max, f_x, f_y, f_z, field_objects='', show=True):
    x, y, z = fc.mesh3d(xyz_max, len(f_x))
    plt.subplot(projection='3d', label='none')
    plt.gca().set_zlabel(r'$z$')
    plt.quiver(x, y, z, f_x, f_y, f_z, length=xyz_max / len(f_x), normalize=True)
    draw_loop() if field_objects == 'loop' else 0
    draw_sphere() if field_objects == 'sphere' else 0
    fa.render_frame(aspect=False) if show else 0


def potential_lines(xy_max, f_xy, field_objects=(None,), show=False):
    x, y, z = fc.mesh3d(xy_max, len(f_xy), indexing='xy')
    plane = round(len(z) / 2)
    f_xy_levels = np.linspace(np.min(f_xy[:, :, plane]) / 10, np.max(f_xy[:, :, plane]) / 10, 4)
    plt.contour(x[:, :, plane], y[:, :, plane], f_xy[:, :, plane], f_xy_levels, colors='k', alpha=0.5)
    draw(field_objects)
    fa.render_frame() if show else 0


def field_lines(xy_max, f_x, f_y, field_objects=(None,), show=True):
    x, y, z = fc.mesh3d(xy_max, len(f_x), indexing='xy')
    plane = round(len(z) / 2)
    f_xy_norm = np.hypot(f_x[:, :, plane], f_y[:, :, plane])
    plt.streamplot(x[:, :, plane], y[:, :, plane], f_x[:, :, plane], f_y[:, :, plane],
                   color=np.log(f_xy_norm), cmap='cool', zorder=0, density=2)
    draw(field_objects)
    fa.render_frame() if show else 0


def draw(field_objects):
    if field_objects[0] is not None:
        for field_object in field_objects:
            field_object.form()


def draw_loop():
    ring = Ellipse((0, 0), 2, 3, edgecolor='black', fill=False)
    plt.gca().add_patch(ring)
    art3d.pathpatch_2d_to_3d(ring, z=0, zdir="x")
    plt.quiver(0., 0., 0., 0., 1., 0., color='red')
    plt.quiver(0., 1., 0., 0., 0., 1.5, color='green')


def draw_sphere():
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    plt.gca().plot_surface(x, y, z, color="r", alpha=.33)
