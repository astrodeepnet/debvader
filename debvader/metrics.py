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
@tf.function
def vae_loss(ground_truth, predicted_distribution):
    """
    computes the reconstruction loss term for the VAE.

    The function takes as input the output of the decoder which is a distribution for each pixel.
    It first calculates the probability of each pixel ground truth under the predicted distrib,
    sums over all the pixels in an image, and finally returns the average over the batch.

    parameters:
        ground_truth: original ground truth image
        predicted_distribution: output distribution from the decoder network
    """
    return -tf.math.reduce_mean(
        tf.math.reduce_sum(
            predicted_distribution.log_prob(ground_truth), axis=[1, 2, 3]
        )
    )
