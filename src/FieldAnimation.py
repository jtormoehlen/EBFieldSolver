import os
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation


def render_frame(x_label='$x$', y_label='$y$', back_color='white', show=True, aspect=True):
    plt.xlabel(r'' + x_label + '')
    plt.ylabel(r'' + y_label + '')
    # plt.rcParams["figure.figsize"] = (5, 5)
    plt.gca().set_facecolor(back_color)
    aspect_ratio(aspect)
    plt.figure().show() if show else 0


def aspect_ratio(aspect=True):
    if aspect:
        plt.gca().set_aspect('equal')
    else:
        plt.gca().set_aspect('auto')


def render_anim(location, t):
    fig = plt.figure()
    frames = []
    for i in range(0, len(t), 1):
        path = 'img/temporary/' + location + str(i) + '.png'
        img = mpimg.imread(path)
        frame = plt.imshow(img, animated=True)
        frames.append([frame])
        os.remove(path)

    animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=50)
    plt.show()


def save_anim(location, t):
    with iio.get_writer('img/dynamic/' + location + '.gif', mode='I') as writer:
        for i in range(0, len(t), 1):
            s = 'img/temporary/' + location + str(i) + '.png'
            image = iio.imread(s)
            writer.append_data(image)
            os.remove(s)
    print('Saving ./dynamic/' + str(location) + '.gif')


def save_frame(location):
    render_frame(show=False)
    location = 'temporary/' + location
    for i in range(0, 100, 1):
        path = 'img/' + location + str(i) + '.png'
        if not os.path.exists(path):
            plt.savefig(path)
            print('Saving ./' + location + '.png#' + str(i + 1))
            break
    plt.cla()
