import numpy as np


def field(xyz, n, fobs, t=0, ffunc='', rc='ij'):
    """
    Superpose fields (f_x(t),f_y(t),f_z(t)) of field objects.
    :param xyz: spatial coords [x1,x2,y1,y2,z1,z2]
    :param n: grid points
    :param fobs: field objects
    :param t: time>=0
    :param ffunc: field function
    :param rc: indexing order
    :return: field (f_x,f_y,f_z)
    """
    x, y, z = mesh(xyz, n, rc=rc)
    Fx, Fy, Fz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for fob in fobs:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    ff = getattr(fob, ffunc)
                    F = np.real(ff(x[i][j][k], y[i][j][k], z[i][j][k], t))
                    Fx[i][j][k] += F[0]
                    Fy[i][j][k] += F[1]
                    Fz[i][j][k] += F[2]
    return Fx, Fy, Fz


def mesh(xyz, n, rc='ij'):
    """
    Mesh object of (x,y,z).
    :param xyz: spatial coords [x1,x2,y1,y2,z1,z2]
    :param n: grid points
    :param rc: indexing order
    :return: mesh
    """
    x, y, z = np.meshgrid(np.linspace(xyz[0], xyz[1], n),
                          np.linspace(xyz[2], xyz[3], n),
                          np.linspace(xyz[4], xyz[5], n),
                          indexing=rc)
    return x, y, z


def field_limit(f, lb):
    """
    Limit arrow length in field.
    :param f: field (Fx,Fy)
    :param lb: lower bound
    :return: limited field
    """
    Fx, Fy = f
    ub = lb * 2.0  # set upper bound to 2*lb
    for i in range(len(Fx)):
        for j in range(len(Fy)):
            fnorm = np.sqrt(Fx[i][j] ** 2 + Fy[i][j] ** 2)
            if fnorm > ub != 0.0:
                Fx[i][j] = (Fx[i][j] / fnorm) * ub
                Fy[i][j] = (Fy[i][j] / fnorm) * ub
    return Fx, Fy


def phi_unit(xyz, n):
    """
    Azimuthal unit vector.
    :param xyz: spatial coords [x1,x2,y1,y2]
    :param n: grid points
    :return: unit vector phi/|phi|
    """
    x, y, z = mesh(xyz, n)
    m = round(n / 2)
    phi = np.arctan2(y[:, :, m], x[:, :, m])
    ephi = np.array([-np.sin(phi), np.cos(phi)])
    return ephi


def grad(F):
    """
    Gradient of scalar field.
    :param F: scalar field
    :return: gradient
    """
    fx, fy, fz = np.gradient(F)
    return np.array([fx, fy, fz])


def rot(Fx, Fy, Fz):
    """
    Curl of vector field.
    :param Fx: x-comp of field
    :param Fy: y-comp of field
    :param Fy: z-comp of field
    :return: curl
    """
    fxx, fxy, fxz = np.gradient(Fx)
    fyx, fyy, fyz = np.gradient(Fy)
    fzx, fzy, fzz = np.gradient(Fz)
    fx = fzy - fyz
    fy = fxz - fzx
    fz = fyx - fxy
    return np.array([fx, fy, fz])
