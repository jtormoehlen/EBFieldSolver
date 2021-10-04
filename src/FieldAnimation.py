import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import FieldPlot as fp


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd='\r'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def render_frame(x_label='$x$', y_label='$y$', back_color='white', show=True, aspect=True):
    plt.xlabel(r'' + x_label + '/m')
    plt.ylabel(r'' + y_label + '/m')
    plt.gca().set_facecolor(back_color)
    plt.gca().set_aspect('equal') if aspect else plt.gca().set_aspect('auto')
    plt.show() if show else 0


def dynamic_field(xy_max, t_max, objects, function, n_xy=30, save=False):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    x, y = fp.init_dynamic(xy_max, n_xy, function)
    f_x, f_y, f_c = fp.dynamic(xy_max, 0, objects, function)
    f_xy_min = np.min(np.sqrt(f_x ** 2 + f_y ** 2))
    f_x, f_y, f_c = fp.dynamic(xy_max, 0, objects, function, f_xy_min)
    Q = ax.quiver(x, y, f_x, f_y,
                  f_c, cmap='cool', pivot='mid')
    ax.plot([0, 0], [-objects.h, objects.h], '-k')
    frames = 20

    def animate(i):
        dt = t_max * (1.0 / frames) * i
        f_x, f_z, f_c = fp.dynamic(xy_max, dt, objects, function, f_xy_min)
        Q.set_UVC(f_x, f_z, f_c)
        printProgressBar(i + 1, frames, prefix='Progress:', suffix='Complete', length=50)
        return Q,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames, interval=100, blit=True)
    plt.tight_layout()

    if save:
        print('Saving ' + function + ' animation...')
        path = 'img/dynamic/' + function + '_' + str(round(objects.L / objects.lambda_0, 3)) + '.gif'
        anim.save(path,
                  writer='imagemagick', fps=10, dpi=100, extra_args=['-layers Optimize'])
        print(path + ' saved')
    else:
        print('Rendering animation...')
        plt.show()
