import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits import mplot3d
import os

from lib.FieldPlot import static, static3d, dynamic, N, draw_antennas

figstylefile=os.path.dirname(__file__)+"/figstyle.mpstyle"
plt.style.use(figstylefile)
# total frames
FRAMES = 30  
# frames per second
FPS = 10  

def window(xy, labs=['$x$', '$y$'], show=True):
    # Set axe limits, labels etc.
    # xy: spatial coords [x1,x2,y1,y2]
    # labs: list<string> of axe labels [x_label,y_label]
    # show: run figure if true
    plt.gca().set_xlim(xy[0], xy[1])
    plt.gca().set_ylim(xy[2], xy[3])
    plt.gca().set_xlabel(labs[0])
    plt.gca().set_ylabel(labs[1])
    if show:
        plt.show()

def static_field(xy, fobs, ffunc, nabla=''):
    # Routine for static field. See FieldPlot.static() for description.
    plt.subplot()
    # extend z-comp
    xy.extend([min(xy),max(xy)])  
    static(xy, fobs, ffunc, nabla)
    window(xy)

def static_field3d(xyz, fobs, ffunc, nabla='', view=''):
    # Routine for 3d-static field. See FieldPlot.static3d() for description.
    plt.subplot(projection='3d', computed_zorder=False)
    static3d(xyz, fobs, ffunc, nabla, view)
    window(xyz)

def dynamic_field(w, t, fobs, ffunc, save=False):
    # Routine for dynamic field. See FieldPlot.dynamic() for description.
    fig = plt.figure()
    plt.subplot()
    # extend z-comp
    w.extend([min(w),max(w)])  
    if ffunc == 'E':
        labels = ['$x$','$z$']
    else:
        labels = ['$x$','$y$']
    # init arrows with avg length
    Q, fmean = dynamic(w, -1, fobs, ffunc)  
    window(w, labels, show=False)

    def init():
        draw_antennas(ffunc, fobs)
        return Q,

    def update(i):
        # current time
        dt = t[0] + t[1]*(i/FRAMES)  
        f_x, f_z, f_c = dynamic(w, dt, fobs, ffunc, fmean)
        # update vectors and colors
        Q.set_UVC(f_x, f_z, f_c)  
        return Q,

    # anim: animator object
    # fig: figure object
    # update: repeat function
    # init: intial function
    # frames: total frames
    # blit: smoothen animation
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=FRAMES, blit=True)

    # save animation
    if save:  
        print('Saving ' + ffunc + ' animation...')
        path = './img/' + ffunc + f'_{round(fobs[0].d/fobs[0].wl, 3)}.gif'
        # writer = FFMpegWriter(fps=FPS) 
        writer = PillowWriter(fps=FPS)  
        anim.save(path, writer=writer,
                  progress_callback=lambda i, j:
                  print(f'Saving frame {i + 1} of {j}'))
    # render animation
    else:  
        print('Rendering animation...')
        plt.show()