import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

path_folder = os.path.dirname(os.path.abspath(__file__))
print(path_folder)
#import debvader
#from debvader import model

###### TO SUPRESSS AND UNCOMMENT PREVIOUS LINES
import sys
sys.path.insert(0,'.')
import model
######

def load_deblender(survey, input_shape, latent_dim, filters, kernels, return_encoder_decoder = False):
    """
    load weights trained for a particular dataset
    parameters:
        survey: string calling the particular dataset (choices are: "dc2")
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """
    # Create the model
    net, encoder, decoder = model.create_model_vae_2(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
    )

    # Define the loss function
    def vae_loss(x, x_decoded_mean):
        #xent_loss = K.mean(
        #    K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1, 2, 3])
        #)
        return -x_decoded_mean.log_prob(x)#xent_loss

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compile the model
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        experimental_run_tf_function=False,
    )

    # Load the weights corresponding to the chosen survey
    loading_path = str(path_folder) + "/../data/weights/" + survey + "/not_normalised/"
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)


    if return_encoder_decoder:
        return net, encoder, decoder
    else:
        return net


def deblend(net, images):
    """
    Deblend the image using the network
    parameters:
        net: network to test
        images: array of images. It can contain only one image.
    """
    # Normalize the images
    #images_normed = np.tanh(np.arcsinh(images))

    # Denorm deblended images
    #images_deblended = np.sinh(np.arctanh(net.predict(images_normed)))

    return net(tf.cast(images, tf.float32)).mean().numpy(), net(tf.cast(images, tf.float32))
