import numpy as np
from matplotlib import pyplot as plt


def plot_arrows(x, y, f_x, f_y, cmap=None, cvalue=0, normalize=False):
    if normalize:
        f_norm = np.hypot(f_x, f_y)
    else:
        f_norm = 1.0
    if cmap is None:
        plt.gca().set_facecolor('white')
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm)
    else:
        plt.gca().set_facecolor('black')
        plt.rcParams['image.cmap'] = cmap
        plt.quiver(x, y, f_x / f_norm, f_y / f_norm, cvalue)


def trim(f_x, f_y, cap):
    for i in range(len(f_x)):
        for j in range(len(f_y)):
            while f_x[i][j] > cap or f_y[i][j] > cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
            while f_x[i][j] < -cap or f_y[i][j] < -cap:
                f_x[i][j] = 0.9 * f_x[i][j]
                f_y[i][j] = 0.9 * f_y[i][j]
    return f_x, f_y


def plot_normal(x, f_x):
    plt.gca().set_facecolor('white')
    plt.plot(x, f_x)


def plot_contour(x, y, f_xy, levels=None):
    plt.gca().set_facecolor('white')
    plt.contour(x, y, f_xy, levels=levels)


def plot_contourf(x, y, f_xy, levels, cmap='bwr'):
    plt.gca().set_facecolor('white')
    plt.contourf(x, y, f_xy, levels, cmap=cmap)


def plot_streamlines(x, y, f_x, f_y, color='black', cmap=None, zorder=2, density=1):
    plt.gca().set_facecolor('white')
    plt.gca().set(xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)))
    plt.streamplot(x, y, f_x, f_y, color=color, cmap=cmap, zorder=zorder, density=density)


def plot_intensity(x, y, f_xy):
    plt.gca().set_facecolor('black')
    plt.pcolormesh(x, y, f_xy, cmap='hot')


def plot_forms(forms):
    plt.gca().set_facecolor('white')
    for f in forms:
        plt.gca().add_patch(f)


def plot_details(details):
    plt.gca().set_facecolor('white')
    for d in details:
        plt.scatter(d[0], d[1], d[2], d[3], d[4], zorder=3)
