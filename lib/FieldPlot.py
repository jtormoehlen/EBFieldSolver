import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d
from lib.FieldCalculator import rot, grad, field, field_limit, mesh, phi_u

# grid points 2d/dyn
N = 50
# grid points 3d
N3D = 10

# Function Arguments.
# xy: Spatial coords (floats) [x1,x2,y1,y2]
# xyz: Spatial coords (floats) [x1,x2,y1,y2,z1,z2]
# fobs: Field objects [o1..on]
# ffunc: Field function (string) {'phi','E','A','B'}
# nabla: Nabla operator (string) {'','rot','-grad'}
# F: Field (ndarray) [Fx,Fy,Fz]
# view: View plane (string) {'','xyz','xy','xz','yz'}
# t: Time (float)
# lb: Lower bound (float)
# plot3d: Draw field object (boolean)

def static(xy, fobs, ffunc, nabla):
    # Stationary 2d-field.
    Fx, Fy, Fz = field(xy, N, fobs, ffunc=ffunc, rc='xy')
    if nabla == '':
        if ffunc == 'phi':
            pot_lines(xy, Fx)
        elif ffunc == 'A':
            pot_lines(xy, Fz)  
        elif ffunc == 'E' or ffunc == 'B':
            field_lines(xy, [Fx,Fy])
    else:
        if nabla == 'rot':
            Fy, Fx, Fz = -rot(Fy, Fx, Fz)
        elif nabla == '-grad':
            Fy, Fx, Fz = -grad(Fx)
        field_lines(xy, [Fx,Fy])
    draw_fobs(ffunc, fobs)

def static3d(xyz, fobs, ffunc, nabla, view):
    # Stationary 3d-field.
    Fx, Fy, Fz = field(xyz, N3D, fobs, ffunc=ffunc)
    if nabla == '':
        if ffunc == 'phi':
            pot_surface(xyz, -Fx)
        elif ffunc == 'A':
            pot_surface(xyz, Fz)
        elif ffunc == 'E' or ffunc == 'B':
            field_arrows3d(xyz, [Fx,Fy,Fz], view)
    else:
        if nabla == 'rot':
            Fx, Fy, Fz = rot(Fx, Fy, Fz)
        elif nabla == '-grad':
            Fx, Fy, Fz = -grad(Fx)
        field_arrows3d(xyz, [Fx,Fy,Fz], view)
    draw_fobs(ffunc, fobs, plot3d=True)

def init_dynamic(xy, fobs, ffunc):
    # Init Time-dependent field.
    n = round(N/2)
    x, y, z = mesh(xy, N)
    if ffunc == 'E':
        x = x[:,n,:]
        y = z[:,n,:]
    elif ffunc == 'B':
        x = x[:,:,n]
        y = y[:,:,n]
    Fx, Fy, Fc = dynamic(xy, 0, fobs, ffunc)
    # Average field strength
    fmean = np.mean(np.hypot(Fx, Fy))
    # Binary colored map (blue,red)
    cmap = plt.cm.get_cmap('coolwarm', 2)  
    Q = plt.gca().quiver(x, y, Fx, Fy, Fc, cmap=cmap, pivot='mid')
    return Q, fmean

def dynamic(xy, t, fobs, ffunc, lb=0.0):
    # Time-dependent field.
    # return: Field and colors [F1,F2,Fc]
    n = round(N/2)
    if t == -1:
        return init_dynamic(xy, fobs, ffunc)
    else:
        Fx, Fy, Fz = field(xy, N, fobs, t, ffunc)
        Fx, Fy, Fz = rot(Fx, Fy, Fz)
        if ffunc == 'E':
            Fx, Fy, Fz = rot(Fx, Fy, Fz)
            field_limit([Fx[:,n,:], Fz[:,n,:]], lb)
            fnorm = np.hypot(Fx[:,n,:], Fz[:,n,:])
            # Color by z-comp
            return Fx[:,n,:], Fz[:,n,:], Fz[:,n,:]/fnorm
        elif ffunc == 'B':
            field_limit([Fx[:,:,n], Fy[:,:,n]], lb)
            fnorm = np.hypot(Fx[:,:,n], Fy[:,:,n])
            phix, phiy, phiz = phi_u(xy, N)
            # Color by phi-comp
            return Fx[:,:,n], Fy[:,:,n], Fx[:,:,n]/fnorm*phix[:,:,n]

def field_arrows3d(xyz, F, view):
    # Plot arrows of 3d-field.
    Fx, Fy, Fz = F
    x, y, z = mesh(xyz, N3D)
    ax = plt.gca()
    n = round(N3D/2)
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_zlabel(r'$z$')
    p = np.zeros_like(z)
    for i in range(len(p)):
        for j in range(len(p)):
            # Only xy-plane 
            if view == 'xy':  
                p[i,j,n] = 1
                ax.view_init(90, -90)
            # Only xz-plane 
            elif view == 'xz':  
                p[i,n,j] = 1
                ax.view_init(0, -90)
            # Only yz-plane 
            elif view == 'yz':  
                p[n,i,j] = 1
                ax.view_init(0, 0)
            # All planes
            elif view == 'xyz':  
                p[i,-1,j] = 1
                p[0,i,j] = 1
                p[i,j,0] = 1
                ax.view_init(22.5, -45)
            # All arrows
            else:  
                p = np.ones_like(z)
                ax.view_init(22.5, -45)
    ax.quiver(x, y, z, Fx*p, Fy*p, Fz*p, color='black',
              arrow_length_ratio=0.5, pivot='middle',
              length=2*max(xyz)/N3D, normalize=True, zorder=0)

def pot_lines(xy, F):
    # Plot contour lines of potential.
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N/2)
    # Define number of levels
    flvl = np.linspace(np.min(F[:,:,n])/10, np.max(F[:,:,n])/10, 4)
    plt.gca().contour(x[:,:,n], y[:,:,n], F[:,:,n],
                      flvl, cmap='coolwarm', zorder=0)

def pot_surface(xy, F):
    # Plot planes of potential.
    x, y, z = mesh(xy, N3D)
    plt.gca().view_init(22.5, -45)
    plt.gca().set_zlim3d(np.min(z), np.max(z))
    plt.gca().set_zlabel(r'$z$')
    n = round(N3D/2)
    plt.gca().plot_surface(x[:,:,n], y[:,:,n], -F[:,:,n], 
                           cmap='coolwarm', zorder=0)

def field_lines(xy, F):
    # Plot lines of field.
    Fx, Fy = F
    x, y, z = mesh(xy, N, rc='xy')
    n = round(N/2)
    plt.gca().streamplot(x[:,:,n], y[:,:,n], Fx[:,:,n], Fy[:,:,n],
                         color='black', zorder=1, linewidth=1)

def draw_fobs(ffunc, fobs, plot3d=False):
    # Draw current loop or discrete charge distribution.
    if ffunc == 'E' or ffunc == 'phi':
        draw_charges(fobs, plot3d)
    elif ffunc == 'B' or ffunc == 'A':
        draw_loop(fobs, plot3d)

def draw_loop(fobs, plot3d):
    # Draw current loop.
    for k in range(len(fobs)):
        r = fobs[k].r0
        if plot3d:
            # Draw line segments in 3d
            for i in range(len(r)):
                if i+1 < len(r):
                        plt.gca().plot([r[i][0],r[i+1][0]],
                                       [r[i][1],r[i+1][1]],
                                       [r[i][2],r[i+1][2]],
                                       color='grey', ls='-',
                                       lw=2, zorder=2)
            # Connect first and last elem
            plt.gca().plot([r[len(r)-1][0],r[0][0]],
                           [r[len(r)-1][1],r[0][1]],
                           [r[len(r)-1][2],r[0][2]],
                           color='grey', ls='-', lw=2, zorder=2)
        else:
            # Draw line segments in 2d
            for i in range(len(r)):
                if i+1 < len(r):
                        plt.gca().plot([r[i][0],r[i+1][0]],
                                       [r[i][1],r[i+1][1]],
                                       color='grey', ls='-',
                                       lw=2, zorder=2)
            # Connect first and last elem
            plt.gca().plot([r[len(r)-1][0], r[len(r)-1][0]],
                           [r[0][1],r[0][1]],
                           color='grey', ls='-',
                           lw=2, zorder=2)

def draw_charges(fobs, plot3d):
    # Draw discrete charge distribution.
    R = 0.25
    cmap = plt.cm.get_cmap('coolwarm')
    for fob in fobs:
        color = cmap(0.0) if fob.q < 0 else cmap(1.0)
         # Draw colored spheres with radius R
        if plot3d: 
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = R * np.cos(u) * np.sin(v)
            y = R * np.sin(u) * np.sin(v)
            z = R * np.cos(v)
            plt.gca().plot_surface(x-fob.r0[0], y-fob.r0[1], z-fob.r0[2],
                                   color=color, zorder=2)
        # Draw colored circles with radius R
        else:  
            circle = plt.Circle((fob.r0[0], fob.r0[1]), R, color=color)
            plt.gca().add_patch(circle)

def draw_antennas(ffunc, fobs):
    for k in range(len(fobs)):
        s = 1.0
        # Draw grey line with gap s
        if ffunc == 'E':
            h = fobs[k].d/2
            plt.gca().plot([fobs[k].r0[0],fobs[k].r0[0]],
                           [fobs[k].r0[2]+s,fobs[k].r0[2]+h], color='grey')
            plt.gca().plot([fobs[k].r0[0],fobs[k].r0[0]],
                           [fobs[k].r0[2]-s,fobs[k].r0[2]-h], color='grey')
        # Draw grey circle
        elif ffunc == 'B':
            circle = plt.Circle((fobs[k].r0[0], fobs[k].r0[1]),
                                s/2, color='grey')
            plt.gca().add_patch(circle)