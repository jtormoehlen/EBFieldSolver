import imageio as iio
import matplotlib.pyplot as plt


def axes(labels=[r'$x$', r'$y$'], limits=[None], size=[10, 10], aspect=True):
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if limits[0] is not None:
        plt.gca().set(xlim=(-limits[0], limits[0]), ylim=(-limits[1], limits[1]))
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')
    plt.rcParams["figure.figsize"] = (size[0], size[1])


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


def background(color):
    plt.gca().set_facecolor(color)
