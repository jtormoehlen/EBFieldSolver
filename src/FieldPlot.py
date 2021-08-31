import numpy as np
import matplotlib.pyplot as plt
import FieldAnimation as fa
import FieldCalculator as fc
import FieldOperator as fo
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits import mplot3d
from matplotlib.patches import Circle, PathPatch, Ellipse


def static_field(xy_max, objects, function, nabla='', n_xy=20):
    if not isinstance(objects, list):
        objects = [objects]
    if nabla == 'rot':
        f_x, f_y, f_z = fc.field3d(xy_max, n_xy, objects, function=function)
        df_x, df_y, df_z = fo.rot(f_x, f_y, f_z)
        z_plane = round(len(f_z) / 2)
        arrow_field(xy_max, df_x[:, :, z_plane], df_y[:, :, z_plane], normalize=True, show=True)
    if nabla == 'grad':
        f_x, f_y, f_z = fc.field3d(xy_max, n_xy, objects, function=function)
        df_x, df_y, df_z = fo.grad(f_x)
        z_plane = round(len(f_z) / 2)
        arrow_field(xy_max, -df_x[:, :, z_plane], -df_y[:, :, z_plane], normalize=True, show=True)
    if nabla == '':
        f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, function=function, indexing='xy')
        field_lines(xy_max, f_x, f_y, field_objects=objects)


def static_field3d(xyz_max, objects, function, nabla='', n_xyz=6):
    if not isinstance(objects, list):
        objects = [objects]
    if nabla == 'rot':
        f_x, f_y, f_z = fc.field3d(xyz_max, n_xyz, objects, function=function)
        f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
        arrow_field3d(xyz_max, f_x, f_y, f_z, field_objects='loop')
    if nabla == 'grad':
        f_x, f_y, f_z = fc.field3d(xyz_max, n_xyz, objects, function=function)
        f_x, f_y, f_z = fo.grad(f_x)
        arrow_field3d(xyz_max, -f_x, -f_y, -f_z, field_objects='sphere')
    if nabla == '':
        f_x, f_y, f_z = fc.field3d(xyz_max, n_xyz, objects, function=function)
        arrow_field3d(xyz_max, f_x, f_y, f_z)


def dynamic_field(xy_max, t_max, objects, function, n_xy=30, save=False):
    if not isinstance(objects, list):
        objects = [objects]
    t = np.linspace(0., t_max, 50)
    for t_i in t:
        if function == 'E':
            f_x, f_y, f_z = fc.field3d(xy_max, n_xy, objects, t=t_i, function='A', nabla='rotrot')
            plane = round(len(f_y) / 2)
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
            x, y, z = fc.mesh3d(xy_max, n_xy)
            fc.field_round(f_x[:, plane, :], f_z[:, plane, :], xy_max, objects[0])
            plt.quiver(x[:, plane, :], z[:, plane, :],
                       f_x[:, plane, :], f_z[:, plane, :],
                       f_z[:, plane, :], cmap='cool')
        if function == 'H':
            f_x, f_y, f_z = fc.field3d(xy_max, n_xy, objects, t=t_i, function='A', nabla='rot')
            plane = round(len(f_z) / 2)
            f_x, f_y, f_z = fo.rot(f_x, f_y, f_z)
            x, y, z = fc.mesh3d(xy_max, n_xy)
            fc.field_round(f_x[:, :, plane], f_y[:, :, plane], xy_max, objects[0])
            f_xy_norm = np.sqrt(f_x[:, :, plane] ** 2 + f_y[:, :, plane] ** 2)
            plt.quiver(x[:, :, plane], y[:, :, plane],
                       f_x[:, :, plane], f_y[:, :, plane],
                       (f_x[:, :, plane] / f_xy_norm) * fc.phi_unit(xy_max, n_xy), cmap='cool')
        if function == 'S':
            f_x, f_y, f_z = fc.field(xy_max, n_xy, objects, t=t_i, function=function, xz_plane=True)
            fc.field_round(f_x, f_z, xy_max, objects[0])
            cfunc = f_x * fc.radius_unit(xy_max, n_xy)
            arrow_field(xy_max, f_x, f_z, cfunc=cfunc, show=False)
        fa.save_frame(function)
    fa.save_anim(function) if save else fa.render_anim(function)


def arrow_field(xy_max, f_x, f_y, normalize=False, cfunc=None, show=True):
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
    plt.quiver(x, y, f_x, f_y)
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
    x, y = fc.mesh(xy_max, len(f_xy), indexing='xy')
    f_xy_levels = np.linspace(np.min(f_xy) / 10, np.max(f_xy) / 10, 4)
    plt.contour(x, y, f_xy, f_xy_levels, colors='k', alpha=0.5)
    draw(field_objects)
    fa.render_frame() if show else 0


def field_lines(xy_max, f_x, f_y, field_objects=(None,), show=True):
    x, y = fc.mesh(xy_max, len(f_x), indexing='xy')
    f_xy_norm = np.hypot(f_x, f_y)
    plt.streamplot(x, y, f_x, f_y, color=np.log(f_xy_norm), cmap='cool', zorder=0, density=2)
    draw(field_objects)
    fa.render_frame() if show else 0


def draw(field_objects):
    if field_objects[0] is not None:
        for field_object in field_objects:
            field_object.form()


def draw_loop():
    ring = Ellipse((0, 0), 1, 2, edgecolor='black', fill=False)
    plt.gca().add_patch(ring)
    art3d.pathpatch_2d_to_3d(ring, z=0, zdir="x")
    plt.quiver(0., 0., 0., 0., .5, 0., color='red')
    plt.quiver(0., .5, 0., 0., 0., 1., color='green')


def draw_sphere():
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    plt.gca().plot_surface(x, y, z, color="r", alpha=.33)
