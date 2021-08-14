import numpy as np
from FieldObject import Antenna
from FieldCalculator import dynamic_field

# antenna with f=500MHz and P=1W
frequency = 500.e6
power = 1.
L_factor = 1./2.
antenna = Antenna(frequency, power, L_factor)


if __name__ == "__main__":
    xyz_max = 2 * antenna.lambda_0
    t_max = antenna.T

    dynamic_field(xyz_max, t_max, antenna, function='E')
    dynamic_field(xyz_max, t_max, antenna, function='H')
    dynamic_field(xyz_max, t_max, antenna, function='S')
