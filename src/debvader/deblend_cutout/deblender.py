import tensorflow as tf

from debvader.normalize.normalize import normalize_non_linear, denormalize_non_linear


def deblend(net, images, normalise=False):
    """
    Deblend the image using the network
    parameters:
        net: neural network used to do the deblending
        images: array of images. It can contain only one image
        normalise: boolean to indicate if images need to be normalised
    """
    if normalise:
        # Normalize input images
        images = normalize_non_linear(images)

    outimg = net(tf.cast(images, tf.float32))

    if normalise:
        # Denorm output images
        outimg = denormalize_non_linear(outimg)

    return (outimg.mean().numpy(), outimg)

