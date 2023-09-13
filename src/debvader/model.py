import os

import numpy as np
import pkg_resources
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.math import fill_triangular
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
tfb = tfp.bijectors


class Normal(tf.Module):
    def __init__(self, input_shape, name=None):
        super().__init__(name=name)
        self.input_shape = input_shape

    def __call__(self, t):
        mean = t[..., : self.input_shape[-1]]
        scale = 1e-4 + t[..., self.input_shape[-1] :]
        sample = tf.random.normal(
            [],
            mean=mean,
            stddev=scale,
        )
        return sample


class MvNormal(tf.Module):
    def __init__(self, latent_dim, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim

    def __call__(self, t):
        diag_shift = np.array(1e-5, t.dtype.as_numpy_dtype)
        scale_tril = fill_triangular(t[..., self.latent_dim :])
        diag = tf.nn.softplus(tf.linalg.diag_part(scale_tril)) + diag_shift
        scale_tril = tf.linalg.set_diag(scale_tril, diag)
        #   scale_tril_bij = tfb.FillScaleTriL(diag_shift=diag_shift)
        #   scale_tril = scale_tril_bij(t[..., self.latent_dim:])

        loc = t[..., : self.latent_dim]
        samples = tf.random.normal([], mean=tf.zeros_like(loc), dtype=t.dtype)
        return loc + tf.linalg.matvec(scale_tril, samples)


def create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
):
    # Define the prior for the latent space
    # TODO: unused in the code, verify if this prior is needed or not
    # prior = tfd.Independent(
    #     tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    # )

    # Input layer
    input_layer = Input(shape=(input_shape))

    # Define the model
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(
            filters[i], (kernels[i], kernels[i]), activation=None, padding="same"
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
    h = Dense(
        tfp.layers.MultivariateNormalTriL.params_size(latent_dim), activation=None
    )(h)

    return Model(input_layer, h)


def create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
    for_onnx=False,
):
    input_layer = Input(shape=(latent_dim,))
    h = PReLU()(input_layer)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32))(h)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0] / 2 ** (len(filters))))
    h = Dense(w * w * filters[-1], activation=dense_activation)(tf.cast(h, tf.float32))
    h = PReLU()(h)
    h = Reshape((w, w, filters[-1]))(h)
    for i in range(len(filters) - 1, -1, -1):
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=conv_activation,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=conv_activation,
            padding="same",
        )(h)
        h = PReLU()(h)

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
    if for_onnx:
        h = Normal(input_shape)(h)
    else:
        h = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Normal(
                loc=t[..., : input_shape[-1]], scale=1e-4 + t[..., input_shape[-1] :]
            ),
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        )(h)

    return Model(input_layer, h)


def create_model_vae(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
    for_onnx=False,
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
        conv_activation=None,
        dense_activation=None,
    )

    decoder = create_decoder(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
        for_onnx=for_onnx,
    )
    # Build the model
    x_input = Input(shape=(input_shape))

    if for_onnx:
        z = MvNormal(latent_dim)(encoder(x_input))
    else:
        # Define the prior for the latent space
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
        )

        z = tfp.layers.MultivariateNormalTriL(
            latent_dim,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01),
        )(encoder(x_input))

    net = Model(inputs=x_input, outputs=decoder(z))

    return net, encoder, decoder, Model(inputs=x_input, outputs=z)


def load_deblender(
    survey,
    input_shape,
    latent_dim,
    filters,
    kernels,
    return_encoder_decoder_z=False,
    for_onnx=False,
):
    """
    load weights trained for a particular dataset
    parameters:
        survey: string calling the particular dataset (choices are: "dc2")
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
        return_encoder_decoder_z: decides whether to return the encoder, decoder, and latent space or not
    """
    # Create the model
    net, encoder, decoder, z = create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
        for_onnx=for_onnx,
    )

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compile the model
    net.compile(
        optimizer=tf.optimizers.legacy.Adam(learning_rate=1e-4),
        loss=vae_loss,
        experimental_run_tf_function=False,
    )

    # Load the weights corresponding to the chosen survey
    data_path = pkg_resources.resource_filename("debvader", "data/")
    loading_path = os.path.join(data_path, "weights", survey, "not_normalised", "loss")
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    if return_encoder_decoder_z:
        return net, encoder, decoder, z
    else:
        return net

