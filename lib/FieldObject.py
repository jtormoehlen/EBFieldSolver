import numpy as np

# vacuum speed of light [cm/s]
c = 2.998E10  

class Charge:
    # Point charge in gaussian cgs-units.
    # q: Charge
    # x: Position x
    # y: Position y
    # z: Position z
    # E: Electric field
    # phi: Electric Potential
    def __init__(self, q, x, y, z=0):
        self.q = q
        self.r0 = np.array([x,y,z])

    def E(self, x, y, z, t=0):
        r = np.array([x,y,z])
        return self.q*(r-self.r0)/np.linalg.norm(r-self.r0)**3

    def phi(self, x, y, z, t=0):
        r = np.array([x,y,z])
        return self.q/np.linalg.norm(r-self.r0)*np.array([1,0,0])

class Current:
    # Current loop in gaussian cgs-units.
    # I: Current strength
    # r0: List of positions [(x1,y1,z1),...,
    #                               (xn-1,yn-1,zn-1)]
    # dr: List of current elements [(drx1,dry1,drz1),...,
    #                                      (drxn-1,dryn-1,drzn-1)]
    # B: Magnetic field
    # A: Vector potential
    def __init__(self, I, r0, dr):
        self.I = I
        self.r0 = r0
        self.dr = dr

    def B(self, x, y, z, t=0):
        r = np.array([x,y,z])
        B = 0
        for i in range(len(self.r0)):
            dl_cross_r_r0 = np.cross(self.dr[i], r-self.r0[i])
            B += 1/c*self.I*dl_cross_r_r0/np.linalg.norm(r-self.r0[i])**3
        return B

    def A(self, x, y, z, t=0):
        r = np.array([x,y,z])
        A = 0
        for i in range(len(self.r0)):
            A += 1/c*self.I*self.dr[i]/np.linalg.norm(r-self.r0[i])
        return A

class Antenna:
    # Antenna in gaussian cgs-units.
    # f: Frequency
    # P: Radiation power
    # n: Antenna length factor d=n*wl
    # x: Position x
    # y: Position y
    # z: Position z
    # E: Electric field
    # B: Magnetic field
    def __init__(self, f, P, n=0, x=0, y=0, z=0, phi=0):
        self.r0 = np.array([x,y,z])
        self.P = P
        self.omega = 2*np.pi*f
        self.wl = c/f
        self.k = 2*np.pi/self.wl
        self.d = n*self.wl/2
        self.phi = np.radians(phi)
        if self.d > 0.0:
            theta, dtheta = np.linspace(0.01, np.pi-0.01, 
                                        100, retstep=True)
            # solve int dtheta*f(theta) with theta in [0,pi]
            self.F = np.trapz(self.__f(theta)**2*np.sin(theta)**3,
                              theta, dtheta)  

    def __f(self, theta):
        return (np.cos(self.k*self.d/2*np.cos(theta))
                - np.cos(self.k*self.d/2))/np.sin(theta)**2

    def E(self, x, y, z, t=0):
        return self.__A(x, y, z, t)*1j/self.k

    def B(self, x, y, z, t=0):
        return self.__A(x, y, z, t)

    def __A(self, x, y, z, t=0):
        r = np.sqrt((x-self.r0[0])**2 
                    + (y-self.r0[1])**2
                    + (z-self.r0[2])**2)
        f_rt = np.exp(1j*(self.k*r - self.omega*t + self.phi))/r
        curr = np.sqrt(c*self.P)
        # short dipole
        if self.d == 0:  
            curr *= np.sqrt(3.0)
            p = 1j*c*curr/self.omega**2*np.array([0,0,1])
            return -1j*self.k*p*f_rt
        # linear antenna
        else:  
            theta = np.arccos((z - self.r0[2])/r)
            curr *= np.sqrt(1/self.F)
            return 2*curr/self.omega*f_rt*self.__f(theta)*np.array([0,0,1])