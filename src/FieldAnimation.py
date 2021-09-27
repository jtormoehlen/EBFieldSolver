import sys

import matplotlib.pyplot as plt
from matplotlib import animation

import FieldPlot as fp


def render_frame(x_label='$x$', y_label='$y$', back_color='white', show=True, aspect=True):
    plt.xlabel(r'' + x_label + '/m')
    plt.ylabel(r'' + y_label + '/m')
    plt.gca().set_facecolor(back_color)
    plt.gca().set_aspect('equal') if aspect else plt.gca().set_aspect('auto')
    plt.show() if show else 0


def render_anim(xy_max, t_max, objects, function, n_xy=30, save=False):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    x, y = fp.init_dynamic_field(xy_max, n_xy, function)
    f_x, f_y, f_c = fp.dynamic_field(xy_max, 0, objects, function)
    Q = ax.quiver(x, y, f_x, f_y, f_c, cmap='cool')

    def animate(i):
        dt = t_max * 0.05 * i
        f_x, f_z, f_c = fp.dynamic_field(xy_max, dt, objects, function)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=20, interval=100, blit=True)
    plt.tight_layout()

    if save:
        print('Rendering .gif...')
        anim.save('img/dynamic/' + function + '.gif',
                  writer='imagemagick', fps=10, dpi=100, extra_args=['-layers Optimize'])
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
