import imageio as iio
import numpy as np
from matplotlib import pyplot as plt


def render_anim(t, loc):
    with iio.get_writer('img/dynamic/' + loc + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/static/' + loc + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)


def render_frame(x_label, y_label, counter, t, loc):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_aspect('equal')
    plt.savefig('img/static/' + loc + str(counter) + '.png')
    plt.cla()
    print(loc + '#' + str(counter + 1) + "/" + str(len(t)))


def plot_arrows(x, y, f_x, f_y, cmap=None, map_z=0.0, normalize=False):
    if normalize:
        f_norm = np.hypot(f_x, f_y)
    else:
        f_norm = 1.0
    if cmap is None:
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm)
    else:
        plt.rcParams['image.cmap'] = cmap
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm, map_z)
