# Import necessary librairies

import numpy as np

import sys
import os

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import  Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten,  Reshape, Cropping2D, Conv2DTranspose, PReLU, Lambda, BatchNormalization
import tensorflow_probability as tfp
tfd = tfp.distributions



def create_model_vae(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation=None, dense_activation=None):
    """
    Create the VAE model
    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        hidden_dim: size of the two dense layers before and after the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """
    # Define the prior for the latent space
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                        reinterpreted_batch_ndims=1)
    
    # Input layer
    input_layer = Input(shape=(input_shape)) 

    # Define the model
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h_2 = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))
    
    h_3 = PReLU()(h_2)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32))(h_3)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='relu', padding='same')(h)

    # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)

    # Generate the model
    model = Model(input_layer,h)
    encoder = Model(input_layer,h_2)
    decoder = Model(h_3,h)

    return model, encoder, decoder

