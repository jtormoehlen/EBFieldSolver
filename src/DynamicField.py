import numpy as np

from FieldObject import Antenna
from FieldAnimation import dynamic_field

"""frequency f=500MHz"""
frequency = 1.0e9

"""radiation power P=1W"""
power = 1.0

"""rod wavelength factor"""
length = 1.5

"""initialize antenna with parameters"""
antenna = Antenna(frequency, power, length)

if __name__ == "__main__":
    """lambda_0 -> wavelength of antenna wave"""
    xyz_max = 2.0 * antenna.lambda_0
    """T -> period of antenna oscillation"""
    t_max = antenna.T

    # print((antenna.k_0 * antenna.L) / np.pi)
    dynamic_field(xyz_max, t_max, antenna, 'E', save=True)
    # dynamic_field(xyz_max, t_max, antenna, 'H', save=True)
