import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from FieldPlot import static, static3d, dynamic
from FieldCalculator import mesh


def init(xy_max, x_label=r'$x$/m', y_label=r'$y$/m', back_color='white', show=True, aspect=True):
    plt.xlim(-xy_max, xy_max)
    plt.ylim(-xy_max, xy_max)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_facecolor(back_color)
    plt.gca().set_aspect('equal') if aspect else plt.gca().set_aspect('auto')
    plt.tight_layout()
    plt.show() if show else 0


def static_field(xy_max, objects, function, nabla=''):
    static(xy_max, objects, function, nabla)
    if function == 'E' or function == 'phi':
        init(xy_max, x_label=r'$x/$d', y_label=r'$y/$d')
    elif function == 'B' or function == 'A':
        init(xy_max, x_label=r'$x/$R', y_label=r'$y/$R')


def static_field3d(xyz_max, objects, function, nabla=''):
    static3d(xyz_max, objects, function, nabla)
    if function == 'E' or function == 'phi':
        init(xyz_max, x_label=r'$x/$d', y_label=r'$y/$d', aspect=False)
    elif function == 'B' or function == 'A':
        init(xyz_max, x_label=r'$x/$R', y_label=r'$y/$R', aspect=False)


def dynamic_field(xy_max, t_max, objects, function, save=False):
    FRAMES = 25
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    x, y, z = mesh(xy_max, 30)
    plane = round(len(z) / 2)
    if function == 'E':
        init(xy_max, x_label=r'$x/\lambda_0$', y_label=r'$z/\lambda_0$', show=False, back_color='black')
        x = x[:, plane, :]
        y = z[:, plane, :]
    if function == 'H':
        init(xy_max, x_label=r'$x/\lambda_0$', y_label=r'$y/\lambda_0$', show=False, back_color='black')
        x = x[:, :, plane]
        y = y[:, :, plane]
    f_x, f_y, f_c = dynamic(xy_max, 0, objects, function)
    f_xy_min = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    f_x, f_y, f_c = dynamic(xy_max, 0, objects, function, f_xy_min)
    ax.set_xlim(-xy_max / objects.lambda_0, xy_max / objects.lambda_0)
    ax.set_ylim(-xy_max / objects.lambda_0, xy_max / objects.lambda_0)
    Q = ax.quiver(x / objects.lambda_0, y / objects.lambda_0, f_x, f_y, f_c, cmap='cool', pivot='mid')

    def update(i):
        dt = t_max * (1.0 / FRAMES) * i
        f_x, f_z, f_c = dynamic(xy_max, dt, objects, function, f_xy_min)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

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
