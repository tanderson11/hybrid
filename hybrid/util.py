import enum
import numpy as np

class RoundingMethod(enum.Enum):
    no_rounding = 'no_rounding'
    conventionally = 'conventionally'
    randomly = 'randomly'

def round_with_method(y, rounding_method, rng=None):
    """Round the state vector `y` either conventionally or randomly."""
    rounding_method = RoundingMethod(rounding_method)
    if round_with_method == RoundingMethod.no_rounding:
        return y
    elif rounding_method == RoundingMethod.randomly:
        return round_randomly(y, rng)
    else:
        return np.round(y)

def round_randomly(y, rng):
    '''Round vector randomly with probability in proportion to decimal value.'''
    # Round up if random float is less than decimal (small decimal ==> rarely round up). Round down otherwise.
    rounded = (rng.random(y.shape) <= (y - np.floor(y))) + np.floor(y)
    # round once more to turn floats-really-close-to-integers into integers
    return np.round(rounded)