import imageio as iio
import matplotlib.pyplot as plt


def axes(x_label=r'$x$', y_label=r'$y$', limit=None, size=10):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if limit is not None:
        plt.gca().set(xlim=(-limit, limit), ylim=(-limit, limit))
    plt.rcParams["figure.figsize"] = (size, size)


def aspect_ratio(aspect=True):
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')


def save_anim(t, loc):
    with iio.get_writer('img/dynamic/' + loc + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/temporary/' + loc + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)

    print('Saving ./dynamic/' + str(loc) + '.gif')


def save_frame(t=[None], loc='default', pos=0):
    if t[0] is not None:
        loc = 'temporary/' + loc
    else:
        loc = 'static/' + loc

    plt.savefig('img/' + loc + str(pos) + '.png')
    plt.cla()

    print('Saving ./' + loc + '.png#' + str(pos + 1) + "/" + str(len(t)))


def background_color(color):
    plt.gca().set_facecolor(color)
