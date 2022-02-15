import numpy as np


def mse(img1, img2):
    return np.mean(np.square(img1 - img2))


# Define the loss as the log likelihood of the distribution on the image pixels
def vae_loss(x, x_decoded_mean):
    return -x_decoded_mean.log_prob(x)
