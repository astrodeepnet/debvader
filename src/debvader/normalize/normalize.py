import numpy as np

def normalize_non_linear(images):
    return np.tanh(np.arcsinh(images))

def denormalize_non_linear(images_normed):
    return np.sinh(np.arctanh(images_normed))

