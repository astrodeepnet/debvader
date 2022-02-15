import os

import pkg_resources
import tensorflow as tf

from debvader import model


def load_deblender(
    survey, input_shape, latent_dim, filters, kernels, return_encoder_decoder_z=False
):
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
    net, encoder, decoder, z = model.create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
    )

    # Define the loss function
    def vae_loss(x, x_decoded_mean):
        # xent_loss = K.mean(
        #    K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1, 2, 3])
        # )
        return -x_decoded_mean.log_prob(x)  # xent_loss

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compile the model
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        experimental_run_tf_function=False,
    )

    # Load the weights corresponding to the chosen survey
    data_path = pkg_resources.resource_filename("debvader", "data/")
    loading_path = os.path.join(data_path, "weights/", survey, "not_normalised/loss/")
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    if return_encoder_decoder_z:
        return net, encoder, decoder, z
    else:
        return net
