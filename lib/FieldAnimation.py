import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from lib.FieldPlot import static, static3d, dynamic, N, draw_antennas
from mpl_toolkits import mplot3d

plt.style.use('./lib/figstyle.mpstyle')
FRAMES = 30  # total frames
FPS = 60  # frames per second


def init_window(xy, labs=['$x$', '$y$'], bgc='white', show=True):
    """
    Set axe limits, labels, background etc.
    :param xy: list<float> of spatial limits [x1,x2,y1,y2]
    :param labs: list<string> of axe labels [x_label,y_label]
    :param bgc: string of background color
    :param show: run figure if true
    """
    plt.gca().set_xlim(xy[0], xy[1])
    plt.gca().set_ylim(xy[2], xy[3])
    plt.gca().set_xlabel(labs[0])
    plt.gca().set_ylabel(labs[1])
    plt.gca().set_facecolor(bgc)
    plt.gca().set_title('(b)')
    if show:
        plt.show()


def static_field(xy, fobs, ffunc, nabla=''):
    """
    Routine for static field.
    :param xy: list of spatial coords [x1,x2,y1,y2]
    :param fobs: list of field objects
    :param ffunc: string of field function in {'phi','E','A','B'}
    :param nabla: string of nabla operator in {''(none),'rot','grad'}
    """
    plt.subplot()
    xy.append(min(xy))
    xy.append(max(xy))
    static(xy, fobs, ffunc, nabla)
    init_window(xy)


def static_field3d(xyz, fobs, ffunc, nabla='', view=''):
    """
    Routine for 3d-static field.
    :param xyz: list of spatial coords [x1,x2,y1,y2,z1,z2]
    :param fobs: list of field objects
    :param ffunc: string of field function
    :param nabla: string of nabla operator {''(none),'rot','grad'}
    :param view: string of view plane {''(none),'xy','xz','yz'}
    """
    plt.gca().subplot(projection='3d')
    static3d(xyz, fobs, ffunc, nabla, view)
    init_window(xyz)


def dynamic_field(xy, tmax, fobs, ffunc, save=False):
    """
    Routine for dynamic field.
    :param xy: list of spatial coords [x1,x2,y1,y2]
    :param tmax: upper bound of time such 0 <= t <= tmax
    :param fobs: list of field objects
    :param ffunc: field function
    :param save: save animation if true
    """
    fig = plt.figure()
    plt.subplot()
    xy.extend([min(xy), max(xy)])
    if ffunc == 'E':
        labels = ['$x/$cm', '$z/$cm']
    else:
        labels = ['$x/$cm', '$y/$cm']
    Q, fmean = dynamic(xy, -1.0, fobs, ffunc)
    init_window(xy, labels, show=False, bgc='white')

    def init():
        draw_antennas(ffunc, fobs)
        return Q,

    def update(i):
        dt = tmax * (1.0 / FRAMES) * i
        f_x, f_z, f_c = dynamic(xy, dt, fobs, ffunc, fmean)
        Q.set_UVC(f_x, f_z, f_c)
        return Q,

    anim = FuncAnimation(fig, update, init_func=init, frames=FRAMES, interval=100, blit=True)

    if save:
        print('Saving ' + ffunc + ' animation...')
        path = './' + ffunc + f'_{round(fobs[0].d / fobs[0].l, 3)}.gif'
        anim.save(path, writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, j: print(f'Saving frame {i + 1} of {j}'))
        sys.exit(0)
    else:
        print('Rendering animation...')
        plt.show()
