import numpy as np
import tensorflow as tf


def mse(img1, img2):
    """
    computes the MSE between 2 numpy arrays (images)

    parameters:
        img1(np.ndarray): first image
        img2(np.nparray): second image
    """
    return np.mean(np.square(img1 - img2))


# Define the loss as the log likelihood of the distribution on the image pixels
def vae_loss(ground_truth, predicted_distribution):
    """
    computes the reconstaruction loss term for the VAE.

    returns the negative log_prob of the ground truth under the predicted distribution

    parameters:
        ground_truth: original ground truth image
        predicted_distribution: distribution predicted by network
    """
    return -tf.math.reduce_sum(
        predicted_distribution.log_prob(ground_truth), axis=[1, 2, 3]
    )
