import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from FieldPlot import static, static3d, dynamic, N_XYZ
from FieldCalculator import mesh

plt.style.use('./figstyle.mpstyle')


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


def init(xy, labs=[r'$x$', r'$y$'], bcg='white', show=True):
    plt.xlim(xy[0], xy[1])
    plt.ylim(xy[2], xy[3])
    plt.xlabel(labs[0])
    plt.ylabel(labs[1])
    plt.gca().set_facecolor(bcg)
    plt.show() if show else 0


def static_field(xy, fobs, ffunc, nabla='', view='xy'):
    static(xy, o_to_olist(fobs), ffunc, nabla, view)
    init(xy)


def static_field3d(xyz, fobs, ffunc, nabla=''):
    static3d(xyz, o_to_olist(fobs), ffunc, nabla)
    init(xyz)


def dynamic_field(xy, t_max, fobs, ffunc, save=False):
    FRAMES = 25
    fobs = o_to_olist(fobs)
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    x, y, z = mesh(xy, N_XYZ)
    n = round(N_XYZ / 2)
    if ffunc == 'E' or ffunc == 'S':
        init(xy, x_label=r'$x/\lambda_0$', y_label=r'$z/\lambda_0$', show=False, bcg='black')
        x = x[:, n, :]
        y = z[:, n, :]
    if ffunc == 'H':
        init(xy, x_label=r'$x/\lambda_0$', y_label=r'$y/\lambda_0$', show=False, bcg='black')
        x = x[:, :, n]
        y = y[:, :, n]
    f_x, f_y, f_c = dynamic(xy, 0, fobs, ffunc)
    f_xy_min = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    f_x, f_y, f_c = dynamic(xy, 0, fobs, ffunc, f_xy_min)
    ax.set_xlim(-xy / fobs[0].lambda_0, xy / fobs[0].lambda_0)
    ax.set_ylim(-xy / fobs[0].lambda_0, xy / fobs[0].lambda_0)
    Q = ax.quiver(x / fobs[0].lambda_0, y / fobs[0].lambda_0, f_x, f_y, f_c, cmap='cool', pivot='mid')

    def update(i):
        dt = t_max * (1.0 / FRAMES) * i
        f_x, f_z, f_c = dynamic(xy, dt, fobs, ffunc, f_xy_min)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    anim = FuncAnimation(fig, update, frames=FRAMES, interval=100, blit=True)

    if save:
        print('Saving ' + ffunc + ' animation...')
        path = './' + ffunc + f'_{round(fobs[0].L / fobs[0].lambda_0, 3)}.gif'
        anim.save(path, writer=PillowWriter(fps=10),
                  progress_callback=lambda i, j: print(f'Saving frame {i+1} of {j}'))
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
