import os
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation


def render_frame(x_label='$x$', y_label='$y$', back_color='white', show=True, aspect=True):
    plt.xlabel(r'' + x_label + '')
    plt.ylabel(r'' + y_label + '')
    plt.gca().set_facecolor(back_color)
    plt.gca().set_aspect('equal') if aspect else plt.gca().set_aspect('auto')
    plt.show() if show else 0


def render_anim(location):
    fig = plt.gcf()
    frames = []
    dpath = 'img/temporary'
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    for i in range(0, 50, 1):
        fpath = dpath + '/' + location + str(i) + '.png'
        img = mpimg.imread(fpath)
        # plt.rcParams["figure.figsize"] = (10, 10)
        frame = plt.imshow(img, animated=True)
        plt.gca().axis('off')
        frames.append([frame])
        os.remove(fpath)
    animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    plt.show()


def save_anim(location):
    dpath = 'img/dynamic'
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    with iio.get_writer(dpath + '/' + location + '.gif', mode='I') as writer:
        for i in range(0, 50, 1):
            fpath = 'img/temporary/' + location + str(i) + '.png'
            image = iio.imread(fpath)
            writer.append_data(image)
            os.remove(fpath)
    print('Saving ' + os.getcwd() + '/img/dynamic/' + str(location) + '.gif')


def save_frame(location):
    render_frame(show=False)
    ftype = location
    dpath = 'img/temporary'
    location = 'img/temporary/' + location
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    for i in range(0, 50, 1):
        fpath = location + str(i) + '.png'
        if not os.path.exists(fpath):
            plt.savefig(fpath)
            print('Loading ' + ftype + ': ' + str(round(((i + 1) / 50.) * 100.)) + '%')
            break
    plt.cla()
