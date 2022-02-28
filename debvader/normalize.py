import numpy as np


def linear_normalization_cosmos(images):
    """
    linear normalization used for cosmos dataset

    parameters:
        images: numpy array to be normalzied.
    """
    return images / 80000


def linear_denormalization_cosmos(images):
    """
    linear denormalization used for cosmos dataset

    parameters:
        images: numpy array to be denormalzied.
    """
    return images * 80000


def non_linear_normalization_cosmos(images):
    """
    non-linear normalization used for cosmos dataset

    parameters:
        images: numpy array to be normalzied.
    """
    return np.tanh(np.arcsinh(images))


def non_linear_denormalization_cosmos(images):
    """
    non-linear denormalization used for cosmos dataset

    parameters:
        images: numpy array to be denormalzied.
    """
    return np.sinh(np.arctanh(images))
