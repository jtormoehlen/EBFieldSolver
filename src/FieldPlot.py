import numpy as np
import matplotlib.pyplot as plt
import FieldAnimation as fa
import mpl_toolkits.mplot3d.art3d as art3d
import FieldCalculator as fc
from mpl_toolkits import mplot3d
from matplotlib.patches import Circle, PathPatch


def arrow_field(xy_max, f_x, f_y, normalize=False, cfunc=None, show=False):
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
    plot(show)


def arrow_field3d(xyz_max, f_x, f_y, f_z, field_objects=True):
    fa.aspect_ratio(False)
    x, y, z = fc.mesh3d(xyz_max, len(f_x))
    plt.subplot(projection='3d', label='none')
    plt.quiver(x, y, z, f_x, f_y, f_z, length=xyz_max / len(f_x), normalize=True)

    if field_objects:
        ring = Circle((0, 0), 1, edgecolor='black', fill=False)
        plt.gca().add_patch(ring)
        art3d.pathpatch_2d_to_3d(ring, z=0, zdir="x")
        plt.quiver(0., 0., 0., 0., 1., 0., color='red')
        plt.quiver(0., 1., 0., 0., 0., 1., color='green')
    plt.show()


def potential_lines(xy_max, f_xy, field_objects=(None,), show=False):
    x, y = fc.mesh(xy_max, len(f_xy))
    f_xy_levels = np.linspace(np.min(f_xy) / 10, np.max(f_xy) / 10, 4)
    plt.contour(x, y, f_xy, f_xy_levels, colors='k', alpha=0.5)
    draw(field_objects)
    plot(show)


def field_lines(xy_max, f_x, f_y, field_objects=(None,), show=True):
    x, y = fc.mesh(xy_max, len(f_x))
    f_xy_norm = np.hypot(f_x, f_y)
    plt.streamplot(x, y, f_x, f_y, color=np.log(f_xy_norm), cmap='cool', zorder=0, density=2)
    draw(field_objects)
    plot(show)


def plot(show):
    fa.render_frame() if show else 0


def draw(field_objects):
    if field_objects[0] is not None:
        for field_object in field_objects:
            field_object.form()
