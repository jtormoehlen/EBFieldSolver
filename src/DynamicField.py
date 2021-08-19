import numpy as np
from FieldObject import Antenna
from FieldPlot import dynamic_field

# antenna with f=500MHz and P=1W
frequency = 500.e6
power = 2.
length = 1./2.
antenna = Antenna(frequency, power, length)


if __name__ == "__main__":
    # lambda_0 -> wavelength of antenna wave
    xyz_max = 2. * antenna.lambda_0
    # T -> period of antenna oscillation
    t_max = antenna.T

    dynamic_field(xyz_max, t_max, antenna, function='E')
    dynamic_field(xyz_max, t_max, antenna, function='H')
    dynamic_field(xyz_max, t_max, antenna, function='S')
