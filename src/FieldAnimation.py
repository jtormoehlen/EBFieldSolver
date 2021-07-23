import imageio as iio
import matplotlib.pyplot as plt


def render_anim(
        t,
        loc):
    with iio.get_writer('img/dynamic/' + loc + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/temporary/' + loc + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)


def render_frame(
        x_label=r'$x$',
        y_label=r'$y$',
        counter=0,
        t=[],
        loc='unknown',
        x_limit=[],
        y_limit=[],
        aspect=True):

    if len(x_limit) > 1:
        plt.gca().set(xlim=(x_limit[0], x_limit[1]), ylim=(y_limit[0], y_limit[1]))
        plt.rcParams["figure.figsize"] = (x_limit[0], y_limit[0])
    else:
        plt.rcParams["figure.figsize"] = (7.5, 7.5)
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')
    if len(t) > 0:
        loc = 'temporary/' + loc
    else:
        loc = 'static/' + loc
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('img/' + loc + str(counter) + '.png')
    plt.cla()
    print(loc + '#' + str(counter + 1) + "/" + str(len(t)))
