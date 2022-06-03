import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from lib.FieldCalculator import rot, grad, field, field_limit, mesh, phi_unit

N = 50  # grid points 2d/dyn
N3D = 10  # grid points 3d


def static(xy, fobs, ffunc, nabla):
    """
    Stationary 2d-field.
    :param xy: spatial coords [x1,x2,y1,y2]
    :param fobs: field objects
    :param ffunc: field function
    :param nabla: nabla operator {''(none),'rot','grad'}
    """
    Fx, Fy, Fz = field(xy, N, fobs, ffunc=ffunc, rc='xy')
    if nabla == 'rot':
        pot_lines(xy, Fz)
        Fy, Fx, Fz = -rot(Fy, Fx, Fz)
    elif nabla == 'grad':
        pot_lines(xy, Fx)
        Fy, Fx, Fz = -grad(Fx)
    field_lines(xy, [Fx, Fy])
    draw_fobs(ffunc, fobs)


def static3d(xyz, fobs, ffunc, nabla, view):
    """
    Stationary 3d-field.
    :param xyz: list of spatial coords [x1,x2,y1,y2,z1,z2]
    :param fobs: list of field objects
    :param ffunc: field function
    :param nabla: nabla operator {''(none),'rot','grad'}
    :param view: view plane {''(none),'xy','xz','yz'}
    """
    Fx, Fy, Fz = field(xyz, N3D, fobs, ffunc=ffunc)
    if nabla == '':
        if ffunc == 'phi':
            pot_surface(xyz, -Fx)
        elif ffunc == 'A':
            pot_surface(xyz, Fz)
    else:
        if nabla == 'rot':
            Fx, Fy, Fz = rot(Fx, Fy, Fz)
        elif nabla == 'grad':
            Fx, Fy, Fz = -grad(Fx)
        field_arrows3d(xyz, [Fx, Fy, Fz], view)
        draw_fobs(ffunc, fobs, plot3d=True)


def dynamic(xy, t, fobs, ffunc, lb=0.0):
    """
    Time-dependent field.
    :param xy: spatial coords [x1,x2,y1,y2]
    :param t: time>=0
    :param fobs: field objects
    :param lb: lower bound
    :return: E- or H-field
    """
    n = round(N / 2)
    if t == -1.0:
        x, y, z = mesh(xy, N)
        if ffunc == 'E':
            x = x[:, n, :]
            y = z[:, n, :]
        elif ffunc == 'B':
            x = x[:, :, n]
            y = y[:, :, n]
        f_x, f_y, f_c = dynamic(xy, 0, fobs, ffunc)
        fmean = np.mean(np.sqrt(f_x ** 2 + f_y ** 2))
        cmap = plt.cm.get_cmap('coolwarm', 2)
        Q = plt.gca().quiver(x, y, f_x, f_y, f_c, cmap=cmap, pivot='mid')
        return Q, fmean
    else:
        Fx, Fy, Fz = field(xy, N, fobs, t, ffunc)
        Fx, Fy, Fz = rot(Fx, Fy, Fz)
        if ffunc == 'E':
            Fx, Fy, Fz = rot(Fx, Fy, Fz)
            field_limit([Fx[:, n, :], Fz[:, n, :]], lb)
            fnorm = np.hypot(Fx[:, n, :], Fz[:, n, :])
            return Fx[:, n, :], Fz[:, n, :], Fz[:, n, :] / fnorm
        elif ffunc == 'B':
            field_limit([Fx[:, :, n], Fy[:, :, n]], lb)
            fnorm = np.hypot(Fx[:, :, n], Fy[:, :, n])
            return Fx[:, :, n], Fy[:, :, n], Fx[:, :, n] / fnorm * phi_unit(xy, N)


def field_arrows3d(xyz, F, view):
    """
    Plot arrows of 3d-field.
    :param xyz: spatial coords [x1,x2,y1,y2]
    :param F: field (Fx,Fy,Fz)
    :param view: view plane {''(none),'xy','xz','yz'}
    """
    Fx, Fy, Fz = F
    x, y, z = mesh(xyz, N3D)
    ax = plt.gca()  # for ortho proj: proj_type='ortho'
    n = round(N3D / 2)
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_zlabel(r'$z$')
    p = np.zeros_like(z)
    for i in range(len(p)):
        for j in range(len(p)):
            if view == 'xy':
                p[i, j, n] = 1.
                ax.view_init(90, -90)
            elif view == 'xz':
                p[i, n, j] = 1.
                ax.view_init(0, -90)
            elif view == 'yz':
                p[n, i, j] = 1.
                ax.view_init(0, 0)
            else:
                # p = np.ones_like(z)
                p[i, -1, j] = 1.
                p[0, i, j] = 1.
                p[i, j, 0] = 1.
                ax.view_init(22.5, -45)
    ax.quiver(x, y, z, Fx * p, Fy * p, Fz * p, color='black', alpha=0.5,
              arrow_length_ratio=0.5, pivot='middle',
              length=2 * max(xyz) / N3D, normalize=True, zorder=0)


def pot_lines(xy, F):
    """
    Plot contour lines of potential.
    :param xy: spatial coords [x1,x2,y1,y2]
    :param F: heights
    """
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N / 2)
    flvl = np.linspace(np.min(F[:, :, n]) / 10, np.max(F[:, :, n]) / 10, 4)
    plt.gca().contour(x[:, :, n], y[:, :, n], F[:, :, n], flvl, colors='k', alpha=0.5)


def pot_surface(xy, F):
    """
    Plot planes of potential.
    :param xy: spatial coords [x1,x2,y1,y2]
    :param F: heights
    """
    x, y, z = mesh(xy, N3D)
    ax = plt.axes(projection='3d')
    ax.view_init(22.5, -45)
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_zlabel(r'$z$')
    n = round(N3D / 2)
    plt.gca().plot_surface(x[:, :, n], y[:, :, n], F[:, :, n], cmap='coolwarm')


def field_lines(xy, F):
    """
    Plot lines of field.
    :param xy: spatial coords [x1,x2,y1,y2]
    :param F: field (Fx,Fy)
    """
    Fx, Fy = F
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N / 2)
    fnorm = np.hypot(Fx[:, :, n], Fy[:, :, n])
    plt.gca().streamplot(x[:, :, n], y[:, :, n], Fx[:, :, n], Fy[:, :, n],
                         color=np.log(fnorm), cmap='binary', zorder=0, density=1)


def draw_fobs(ffunc, fobs, plot3d=False):
    """
    Draw current loop or discrete charge distribution.
    :param ffunc: field function
    :param fobs: field objects
    :param plot3d: plot 3d if true
    """
    if ffunc == 'E' or ffunc == 'phi':
        draw_charges(fobs, plot3d)
    elif ffunc == 'B' or ffunc == 'A':
        draw_loop(fobs, plot3d)


def draw_loop(fobs, plot3d):
    """
    Draw current loop.
    :param plot3d: plot 3d if true
    :param fobs: field objects
    """
    for k in range(len(fobs)):
        r = fobs[k].r0
        if plot3d:
            for i in range(len(r)):
                for j in range(0, 3, 1):
                    if i + 1 < len(r):
                        plt.gca().plot([r[i][0], r[i + 1][0]],
                                       [r[i][1], r[i + 1][1]],
                                       [r[i][2], r[i + 1][2]],
                                       color='black', ls='-', lw=2., zorder=2)
            plt.gca().plot([r[len(r) - 1][0], r[0][0]],
                           [r[len(r) - 1][1], r[0][1]],
                           [r[len(r) - 1][2], r[0][2]],
                           color='black', ls='-', lw=2., zorder=2)

        else:
            for i in range(len(r)):
                for j in range(0, 2, 1):
                    if i + 1 < len(r):
                        plt.gca().plot([r[i][0], r[i + 1][0]], [r[i][1], r[i + 1][1]],
                                       color='black', ls='-', lw=2., alpha=0.5, zorder=2)
            plt.gca().plot([r[len(r) - 1][0], r[len(r) - 1][0]], [r[0][1], r[0][1]],
                           color='black', ls='-', lw=2., alpha=0.5, zorder=2)


def draw_charges(fobs, plot3d):
    """
    Draw discrete charge distribution.
    :param plot3d: plot 3d if true
    :param fobs: field objects
    """
    R = 0.25
    if plot3d:
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            plt.gca().plot_surface(x - fob.r0[0], y - fob.r0[1], z - fob.r0[2],
                                   color=color, zorder=2)
    else:
        for fob in fobs:
            color = 'blue' if fob.q < 0 else 'red'
            circle = plt.Circle((fob.r0[0], fob.r0[1]), R, color=color)
            plt.gca().add_patch(circle)


def draw_antennas(ffunc, fobs):
    for k in range(len(fobs)):
        s = 1.0
        if ffunc == 'E':
            h = fobs[k].d / 2
            plt.gca().plot([fobs[k].r0[0], fobs[k].r0[0]],
                           [fobs[k].r0[2] + s, fobs[k].r0[2] + h], color='grey')
            plt.gca().plot([fobs[k].r0[0], fobs[k].r0[0]],
                           [fobs[k].r0[2] - s, fobs[k].r0[2] - h], color='grey')
        elif ffunc == 'B':
            circle = plt.Circle((fobs[k].r0[0], fobs[k].r0[1]), s / 2, color='grey')
            plt.gca().add_patch(circle)
