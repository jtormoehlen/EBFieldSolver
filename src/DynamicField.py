from FieldObject import Antenna
from FieldAnimation import render_anim

"""frequency f=500MHz"""
frequency = 500.0e6

"""radiation power P=1W"""
power = 2.0

"""rod wavelength factor"""
length = 3/2

"""initialize antenna with parameters"""
antenna = Antenna(frequency, power, length)


if __name__ == "__main__":
    """lambda_0 -> wavelength of antenna wave"""
    xyz_max = 2 * antenna.lambda_0
    """T -> period of antenna oscillation"""
    t_max = antenna.T

    render_anim(xyz_max, t_max, antenna, 'H', save=True)
