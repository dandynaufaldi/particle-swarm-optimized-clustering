import numpy


def normalize(x: numpy.ndarray):
    """Scale to 0-1
    """
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def standardize(x: numpy.ndarray):
    """Scale to zero mean unit variance
    """
    return (x - x.mean(axis=0)) / numpy.std(x, axis=0)
