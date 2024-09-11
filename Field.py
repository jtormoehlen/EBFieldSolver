import numpy as np

from lib.FieldAnimation import static_field, static_field3d, dynamic_field
from lib.FieldObject import Charge, Current, Antenna

def main():
    # Example: Electrical quadrupole
    # charge [Q]
    q = 1.0  
    # x,y [m] 
    x = y = 1.0
    # -q at (-1,1), q at (1,1), q at (-1,-1), -q at (1,-1)
    charges = [
        Charge(-q, -x, y),
        Charge(q, x, y),
        Charge(q, -x, -y),
        Charge(-q, x, -y)
    ]
    # static_field([-4,4,-4,4], charges, nabla='-grad', ffunc='phi')
    # static_field3d([-2,2,-2,2,-2,2], charges, 
    #                nabla='-grad', ffunc='phi', view='xyz')
    
    # Example: Elliptical conductor loop
    # current [I0]
    I = 1.0
    # current elements positions
    r = []
    # current element directions
    dr = []
    # current elements positions
    # semi-minor axis [m]
    a = 5.0
    # semi-major axis [m]
    b = 15.0
    # number of current elements
    n = 25
    phi = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    dphi = 2*np.pi/n
    for phi_i in phi:
        r_y = a*np.cos(phi_i)
        r_z = b*np.sin(phi_i)
        dr_y = -a*np.sin(phi_i)*dphi
        dr_z = b*np.cos(phi_i)*dphi
        # ..add r_i to r
        r.append(np.array([0,r_y,r_z]))
        # ..add dr_i to dr
        dr.append(np.array([0,dr_y,dr_z]))
    currents = [
        Current(I, r, dr)
    ]
    # static_field([-40,40,-40,40], currents, nabla='rot', ffunc='A')
    # static_field3d([-20,20,-20,20,-20,20], 
    #                currents, nabla='rot', ffunc='A', view='xyz')
    
    # Example: Linear antenna
    # frequency [Hz]
    f = 2.0E9
    # radiation power [W]
    P = 1.0
    antenna = Antenna(f, P, 1.0)
    antennas = [
        antenna
    ]
    # wave length [cm]
    wl = antenna.wl  
    # time span [0,T] with period T=1/f
    t = [0,1/f]
    dynamic_field([-2*wl,2*wl,-2*wl,2*wl], t, 
                  antennas, ffunc='E', save=False)
    # dynamic_field([-2*wl,2*wl,-2*wl,2*wl], t, 
    #               antennaList, ffunc='E', save=True)

if __name__ == "__main__":
    main()