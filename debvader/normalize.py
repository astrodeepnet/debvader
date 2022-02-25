import numpy as np


def linear_normalization_cosmos(x, direction="normalize"):
    """
    linear normalization used for cosmos dataset

    parameters:
        x: numpy array to be normalzied.
        derection: options - "normalize", "denormalize"
    """
    if direction not in ["normalize, denormalize"]:
        raise ValueError(
            'the possible options for direction is either "normalize" or "denormalize"'
        )

    if direction == "normalize":
        return x / 80000
    else:
        return x * 80000


def non_linear_normalization_cosmos(images, direction="normalize"):
    if direction not in ["normalize, denormalize"]:
        raise ValueError(
            'the possible options for direction is either "normalize" or "denormalize"'
        )
    if direction == "normalize":
        # Normalize input images
        images_normed = np.tanh(np.arcsinh(images))
    else:
        images = np.sinh(np.arctanh(images_normed))
