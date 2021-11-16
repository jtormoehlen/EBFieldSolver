import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import FieldPlot as fp

FRAMES = 25


def render_frame(x_label='$x$', y_label='$y$', back_color='black', show=True, aspect=True):
    plt.xlabel(r'' + x_label + '/m')
    plt.ylabel(r'' + y_label + '/m')
    plt.gca().set_facecolor(back_color)
    plt.gca().set_aspect('equal') if aspect else plt.gca().set_aspect('auto')
    plt.show() if show else 0


def dynamic_field(xy_max, t_max, objects, function, save=False):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)

    x, y = fp.init_dynamic(xy_max, function)
    f_x, f_y, f_c = fp.dynamic(xy_max, 0, objects, function)
    f_xy_min = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    f_x, f_y, f_c = fp.dynamic(xy_max, 0, objects, function, f_xy_min)
    Q = ax.quiver(x, y, f_x, f_y,
                  f_c, cmap='cool', pivot='mid')

    def update(i):
        dt = t_max * (1.0 / FRAMES) * i
        f_x, f_z, f_c = fp.dynamic(xy_max, dt, objects, function, f_xy_min)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update,
                                   frames=FRAMES, interval=100, blit=True)

    if save:
        print('Saving ' + function + ' animation...')
        path = 'img/dynamic/' + function + f'_{round(objects.L / objects.lambda_0, 3)}.gif'
        anim.save(path, writer='imagemagick',
                  fps=10, dpi=100, extra_args=['-layers Optimize'],
                  progress_callback=lambda i, n: print(f'Saving frame {i+1} of {n}'))
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
