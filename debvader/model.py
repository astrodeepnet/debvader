import os

import numpy as np
import pkg_resources
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dense,
    Flatten,
    Input,
    PReLU,
    Reshape,
)
from tensorflow.keras.models import Model

from debvader.metrics import vae_loss

tfd = tfp.distributions


def create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
):
    """
    Create the encoder of VAE model

    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """

    # Input layer
    input_layer = Input(shape=(input_shape))

    # Define the model
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
        )(h)
        h = PReLU()(h)
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(1024)(h)
    h = PReLU()(h)
    h = Dense(
        tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
        activation=None,
    )(h)

    return Model(input_layer, h, name="encoder")


def create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
):
    """
    Create the decoder of VAE model

    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """
    input_layer = Input(shape=(latent_dim,))
    h = Dense(256)(input_layer)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0] / 2 ** (len(filters))))
    h = Dense(w * w * filters[-1], activation=None)(tf.cast(h, tf.float32))
    h = PReLU()(h)
    h = Reshape((w, w, filters[-1]))(h)
    for i in range(len(filters) - 1, -1, -1):
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
        )(h)
        h = PReLU()(h)

    # keep the output of the last layer as relu as we want only positive flux values.
    h = Conv2D(input_shape[-1] * 2, (3, 3), activation="relu", padding="same")(h)

    # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
    cropping = int(h.get_shape()[1] - input_shape[0])
    if cropping > 0:
        print("in cropping")
        if cropping % 2 == 0:
            h = Cropping2D(cropping / 2)(h)
        else:
            h = Cropping2D(
                ((cropping // 2, cropping // 2 + 1), (cropping // 2, cropping // 2 + 1))
            )(h)

    # Build the encoder only
    h = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Normal(
            loc=t[..., : input_shape[-1]], scale=1e-3 + t[..., input_shape[-1] :]
        ),
        convert_to_tensor_fn=tfp.distributions.Distribution.sample,
    )(h)

    return Model(input_layer, h, name="decoder")


def create_model_vae(
    input_shape,
    latent_dim,
    filters,
    kernels,
):
    """
    Create the VAE model

    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """

    encoder = create_encoder(
        input_shape,
        latent_dim,
        filters,
        kernels,
    )

    decoder = create_decoder(
        input_shape,
        latent_dim,
        filters,
        kernels,
    )

    # Define the prior for the latent space
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    )

    # Build the model
    x_input = Input(shape=(input_shape))
    z = tfp.layers.MultivariateNormalTriL(
        latent_dim,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01),
        name="latent_space",
    )(encoder(x_input))

    net = Model(inputs=x_input, outputs=decoder(z))

    return net, encoder, decoder, Model(inputs=x_input, outputs=z)


def load_deblender(
    input_shape,
    latent_dim,
    filters,
    kernels,
    return_encoder_decoder_z=False,
    survey="dc2",
    loading_path=None,
):
    """
    load weights trained for a particular dataset
    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
        return_encoder_decoder_z: decides whether to return the encoder, decoder, and latent space or not
        survey: string calling the particular dataset (currently only "dc2" is supported)
        loading_path: optional path to weights (default is None)
            if a path is provided, the `survey` argument will be ignored.
            if left as None, internal weights for the specified `survey` argument will be loaded.
    """
    # Create the model
    net, encoder, decoder, z = create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
    )

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compile the model
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        experimental_run_tf_function=False,
    )

    if loading_path is None:
        # Load the weights corresponding to the chosen survey
        data_path = pkg_resources.resource_filename("debvader", "data/")
        loading_path = os.path.join(
            data_path, "weights", survey, "not_normalised", "loss"
        )

    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    print("weights loaded from: " + loading_path)
    if return_encoder_decoder_z:
        return net, encoder, decoder, z

    return net
