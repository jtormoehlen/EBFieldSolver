import os
import imageio as iio
import matplotlib.pyplot as plt


def show_frame(x_label='$x$', y_label='$y$', back_color='white', location=''):
    plt.xlabel(r'' + x_label + '')
    plt.ylabel(r'' + y_label + '')
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.gca().set_facecolor(back_color)
    if location == '':
        plt.figure().show()


def aspect_ratio(aspect=True):
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')


def save_anim(t, location):
    with iio.get_writer('img/dynamic/' + location + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/temporary/' + location + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)
            os.remove(s)

    print('Saving ./dynamic/' + str(location) + '.gif')


def save_frame(t=[None], location='default', index=0):
    if t[0] is not None:
        location = 'temporary/' + location
    else:
        location = 'static/' + location
    plt.savefig('img/' + location + str(index) + '.png')
    plt.cla()

    print('Saving ./' + location + '.png#' + str(index + 1) + "/" + str(len(t)))
