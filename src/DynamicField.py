from FieldObject import Antenna
from FieldAnimation import dynamic_field

# frequency f=1GHz
frequency = 1.0E9

# radiation power P=1W
power = 1.0

# rod wavelength factor
length_factor = 1.5

# initialize antenna with parameters
antenna = Antenna(frequency, power, length_factor)

if __name__ == "__main__":
    # lambda_0 : wavelength of antenna wave
    xyz_max = 2.0 * antenna.lambda_0
    # T : period of antenna oscillation
    t_max = antenna.T

    dynamic_field(xyz_max, t_max, antenna, function='E', save=True)
    # dynamic_field(xyz_max, t_max, antenna, function='H', save=True)
