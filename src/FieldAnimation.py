import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from FieldPlot import static, static3d, dynamic, N_XYZ
from FieldCalculator import mesh

plt.style.use('./figstyle.mpstyle')


def init(xy, labs=['$x$', '$y$'], bcg='white', show=True):
    plt.xlim(xy[0], xy[1])
    plt.ylim(xy[2], xy[3])
    plt.xlabel(labs[0])
    plt.ylabel(labs[1])
    plt.gca().set_facecolor(bcg)
    plt.show() if show else 0


def static_field(xy, fobs, ffunc, nabla=''):
    xy.append(min(xy))
    xy.append(max(xy))
    static(xy, fobs, ffunc, nabla)
    init(xy)


def static_field3d(xyz, fobs, ffunc, nabla='', view=''):
    static3d(xyz, fobs, ffunc, nabla, view)
    init(xyz)


def dynamic_field(xy, t_max, fobs, ffunc, save=False):
    FRAMES = 25
    FPS = 10
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    xy.append(min(xy))
    xy.append(max(xy))
    x, y, z = mesh(xy, N_XYZ)
    n = round(N_XYZ / 2)
    if ffunc == 'E':
        init(xy, ['$x$', '$z$'], show=False, bcg='black')
        x = x[:, n, :]
        y = z[:, n, :]
    if ffunc == 'H':
        init(xy, ['$x$', '$y$'], show=False, bcg='black')
        x = x[:, :, n]
        y = y[:, :, n]
    f_x, f_y, f_c = dynamic(xy, 0, fobs, ffunc, 1.)
    fmin = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    f_x, f_y, f_c = dynamic(xy, 0, fobs, ffunc, fmin)
    Q = ax.quiver(x, y, f_x, f_y, f_c, cmap='cool', pivot='mid')

    def update(i):
        dt = t_max * (1.0 / FRAMES) * i
        f_x, f_z, f_c = dynamic(xy, dt, fobs, ffunc, fmin)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    anim = FuncAnimation(fig, update, frames=FRAMES, interval=100, blit=True)

    if save:
        print('Saving ' + ffunc + ' animation...')
        path = './' + ffunc + f'_{round(fobs[0].L / fobs[0].lambda_0, 3)}.gif'
        anim.save(path, writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, j: print(f'Saving frame {i+1} of {j}'))
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
