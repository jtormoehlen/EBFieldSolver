import imageio as iio
import numpy as np
from matplotlib import pyplot as plt


def render_anim(t, loc):
    with iio.get_writer('img/dynamic/' + loc + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/static/' + loc + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)


def render_frame(x_label, y_label, counter, t, loc, x_limit=[], y_limit=[], aspect=True):
    if len(x_limit) > 1:
        plt.gca().set(xlim=(x_limit[0], x_limit[1]), ylim=(y_limit[0], y_limit[1]))
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('img/static/' + loc + str(counter) + '.png')
    plt.cla()
    print(loc + '#' + str(counter + 1) + "/" + str(len(t)))


def plot_arrows(x, y, f_x, f_y, cmap=None, map_z=1.0, normalize=False, cap=0):
    if normalize:
        f_norm = np.hypot(f_x, f_y)
    else:
        f_norm = 1.0
    if cap > 0:
        f_x, f_y = trim(f_x, f_y, cap)
    if cmap is None:
        plt.gca().set_facecolor('white')
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm)
    else:
        plt.gca().set_facecolor('black')
        plt.rcParams['image.cmap'] = cmap
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm, np.hypot(f_x, f_y))


def trim(f_x, f_y, cap):
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            if f_x[i][j] > cap:
                f_x[i][j] = cap
            if f_x[i][j] < -cap:
                f_x[i][j] = -cap
            if f_y[i][j] > cap:
                f_y[i][j] = cap
            if f_y[i][j] < -cap:
                f_y[i][j] = -cap
    return f_x, f_y


def plot_normal(x, f_x):
    plt.gca().set_facecolor('white')
    plt.plot(x, f_x)


def plot_contour(x, y, z):
    plt.gca().set_facecolor('white')
    plt.contour(x, y, z, levels=np.linspace(np.min(z), np.max(z), 10))


def plot_intensity(x, y, z):
    plt.gca().set_facecolor('black')
    plt.pcolormesh(x, y, z, cmap='hot')
