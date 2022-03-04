import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from lib.FieldPlot import static, static3d, dynamic, N
from lib.FieldCalculator import mesh

plt.style.use('./lib/figstyle.mpstyle')
FRAMES = 30
FPS = 60


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


def dynamic_field(xy, tmax, fobs, ffunc, save=False):
    fig = plt.figure()
    ax = plt.subplot()
    xy.extend([min(xy), max(xy)])
    x, y, z = mesh(xy, N)
    n = round(N / 2)
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
    Q = ax.quiver(x, y, f_x, f_y, f_c, cmap='cool', pivot='mid')

    def update(i):
        dt = tmax * (1.0 / FRAMES) * i
        f_x, f_z, f_c = dynamic(xy, dt, fobs, ffunc, fmin)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    anim = FuncAnimation(fig, update, frames=FRAMES, interval=100, blit=True)

    if save:
        print('Saving ' + ffunc + ' animation...')
        path = './' + ffunc + f'_{round(fobs[0].L / fobs[0].lambda0, 3)}.gif'
        anim.save(path, writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, j: print(f'Saving frame {i + 1} of {j}'))
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
