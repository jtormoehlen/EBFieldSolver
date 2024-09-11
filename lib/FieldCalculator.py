import numpy as np

def field(xyz, n, fobs, t=0, ffunc='', rc='ij'):
    # Superpose fields of field objects.
    # xyz: Spatial coords [x1,x2,y1,y2,z1,z2]
    # n: Grid points
    # fobs: Field objects
    # t: Time
    # ffunc: Field function
    # rc: Indexing order
    # return: Field [Fx,Fy,Fz]
    x, y, z = mesh(xyz, n, rc=rc)
    Fx, Fy, Fz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
    for fob in fobs:
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    ff = getattr(fob, ffunc)
                    # compute real part Re{[Fx,Fy,Fz]}
                    F = np.real(ff(x[i][j][k], y[i][j][k], z[i][j][k], t))
                    Fx[i][j][k] += F[0]
                    Fy[i][j][k] += F[1]
                    Fz[i][j][k] += F[2]
    return Fx, Fy, Fz

def mesh(xyz, n, rc='ij'):
    # Mesh of spatial coords.
    # xyz: Spatial coords [x1,x2,y1,y2,z1,z2]
    # n: Grid points
    # rc: Indexing order: row -> column
    # return: Mesh
    x, y, z = np.meshgrid(np.linspace(xyz[0], xyz[1], n),
                          np.linspace(xyz[2], xyz[3], n),
                          np.linspace(xyz[4], xyz[5], n),
                          indexing=rc)
    return x, y, z

def field_limit(f, lb):
    # Limit arrow length in field. 
    # f: Field (Fx,Fy)
    # lb: Lower bound
    # return: Limited field
    Fx, Fy = f
    # set upper bound to match visibility
    ub = 2*lb  
    for i in range(len(Fx)):
        for j in range(len(Fy)):
            # vector lengths
            fnorm = np.hypot(Fx[i][j], Fy[i][j])  
            if fnorm > ub != 0.0:
                # limit Fx
                Fx[i][j] = (Fx[i][j]/fnorm)*ub
                # limit Fy
                Fy[i][j] = (Fy[i][j]/fnorm)*ub  
    return Fx, Fy

def phi_u(xyz, n):
    # Azimuthal unit vector.
    # xyz: Spatial coords [x1,x2,y1,y2]
    # n: Grid points
    # return: Unit vector phi/|phi|
    x, y, z = mesh(xyz, n)
    phi = np.arctan2(y, x)
    ephix, ephiy, ephiz = np.array([-np.sin(phi),np.cos(phi),
                                    np.zeros_like(z)])
    return ephix, ephiy, ephiz

def grad(F):
    # Gradient of scalar field.
    # F: Scalar field
    # return: Gradient
    fx, fy, fz = np.gradient(F)
    return np.array([fx,fy,fz])

def rot(Fx, Fy, Fz):
    # Curl of vector field.
    # Fx: x-comp of field
    # Fy: y-comp of field
    # Fy: z-comp of field
    # return: curl
    fxx, fxy, fxz = np.gradient(Fx)
    fyx, fyy, fyz = np.gradient(Fy)
    fzx, fzy, fzz = np.gradient(Fz)
    fx = fzy-fyz
    fy = fxz-fzx
    fz = fyx-fxy
    return np.array([fx,fy,fz])