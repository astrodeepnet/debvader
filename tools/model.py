# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import random
import cmath as cm
import math
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import LocallyConnected2D, Conv2D, Conv3D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU, GlobalAveragePooling2D
import tensorflow as tf
tfd = tfp.distributions


#from keras_gcnn.layers import GConv2D, GroupPool

sys.path.insert(0,'../../scripts/tools_for_VAE/')
from tools_for_VAE import ktied_distribution

# Probabilistic models

#import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.layers import util as tfp_layers_util
# Weights initialization for posteriors
def get_posterior_fn():
  return tfp_layers_util.default_mean_field_normal_fn(
      loc_initializer=tf.keras.initializers.Henormal(), 
      untransformed_scale_initializer=tf.keras.initializers.RandomNormal(
          mean=-9, stddev=0.1)#mean=-9, stddev=0.1)
      )
# kernel divergence weight in loss
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / (40000))


def create_model_full_prob_rt(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):

    input_layer = Input(shape=(input_shape)) 
    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', 
                                            strides=(2,2))(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = tfp.layers.DenseReparameterization(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=dense_activation)(h)

    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model(input_layer,h)
    
    return model

def create_model_full_prob_flipout(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    
    input_layer = Input(shape=(input_shape)) 
    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(), 
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)


    model = Model(input_layer,h)
    
    return model






# Model with coordinate of target galaxy
def create_model_wo_ls_peak_inv(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    h = GConv2D(filters[0], (kernels[0],kernels[0]), activation=None, padding='same',h_input='Z2',h_output='C4')(h)
    h_2 = GConv2D(filters[0], (kernels[0],kernels[0]), activation=None, padding='same',h_input='Z2',h_output='C4')(h_2)
    for i in range(len(filters)):
        h_2 = GConv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same',h_input='C4',h_output='C4')(h_2)
        #h_2 = PReLU()(h_2)
        h = GConv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same',h_input='C4',h_output='C4')(h)
        #h = PReLU()(h)

        h = GConv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2),h_input='C4',h_output='C4')(h)
        h = PReLU()(h)
        h_2 = GConv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2),h_input='C4',h_output='C4')(h_2)
        h_2 = PReLU()(h_2)

        h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = GroupPool(h_input='C4')(h)
    #h = GlobalAveragePooling2D()(h)

    h = Flatten()(h)
    h = PReLU()(h)

    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h) #Dense(2)(h)#
    #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.2))(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model



def create_model_wo_ls_peak_pooling_dense(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    #h = input_layer_1
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = MaxPool2D()(h)
        h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)

    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    h = Flatten()(tf.cast(h,tf.float64))
    h = Dense(hidden_dim, activation = 'tanh')(h)
    #h = PReLU()(h)
    h = Dense(hidden_dim, activation = 'tanh')(h)
    #h = PReLU()(h)
    h = Dense(hidden_dim)(h)
    h = PReLU()(h)
    h = Dense(int(hidden_dim/2))(h)
    h = PReLU()(h)

    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(tf.cast(h,tf.float32))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    model = Model([input_layer_1, input_layer_2],h)
    return model


def create_model_no_psf_dense(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    #h = input_layer_1
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', use_bias=False)(h)
        h = PReLU()(h)
        h = MaxPool2D()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', use_bias=False)(h)
        h = PReLU()(h)
 
    h = Flatten()(tf.cast(h,tf.float64))
    h = Dense(hidden_dim, activation = 'tanh')(h)
    #h = PReLU()(h)
    h = Dense(hidden_dim, activation = 'tanh')(h)
    #h = PReLU()(h)
    h = Dense(hidden_dim)(h)
    h = PReLU()(h)
    h = Dense(int(hidden_dim/2))(h)
    h = PReLU()(h)

    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(tf.cast(h,tf.float32))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    model = Model([input_layer_1, input_layer_2],h)
    return model

encoded_size = 10
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
def create_model_wo_ls_peak_pooling_concat_one(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    #h = input_layer_1
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = MaxPool2D()(h)
        h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)

    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    h = Flatten()(tf.cast(h,tf.float64))
    #h = Dense(tfp.layers.MultivariateNormalTriL.params_size(encoded_size),activation=None)(h)
    #h = tfp.layers.MultivariateNormalTriL(encoded_size,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))

    #h = PReLU()(h)
    #h = Dense(encoded_size)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(tf.cast(h,tf.float32))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_cyrille(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        l1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', use_bias=False)
        h = l1(h)
        h_2 = l1(h_2)

        h = PReLU()(h)
        h_2 = PReLU()(h_2)

        h = MaxPool2D()(h)
        h_2 = MaxPool2D()(h_2)

        l2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', use_bias=False)
        h = l2(h)
        h_2 = l2(h_2)

        h = PReLU()(h)
        h_2 = PReLU()(h_2)

    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)

    h = Flatten()(tf.cast(h,tf.float64))
    h = Dense(hidden_dim, activation = 'tanh')(h)#, activation = 'tanh'
    #h = PReLU()(h)
    h = Dense(hidden_dim, activation = 'tanh')(h)#, activation = 'tanh'
    #h = PReLU()(h)
    h = Dense(hidden_dim)(h)
    h = PReLU()(h)
    h = Dense(int(hidden_dim/2))(h) #
    h = PReLU()(h) #
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(tf.cast(h,tf.float32))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model




#@tf.function
def shear_img(img):
    shears = [(0.01,0), (-0.01,0), (0,0.01), (0,-0.01)]
    images = K.get_value(img)

    image_sheared = np.zeros((4,len(images[:,0,0,0]),59,59,6))
    for z, input_shear in enumerate(shears):
        for i in range (len(images[:,0,0,0])):
            for k in range (len(images[0,0,0])):
                psf_image = galsim.Image(psf[i,:,:,k], copy=True)
                galsim_psf = galsim.InterpolatedImage(psf_image,scale = 0.2)
                inv_psf = galsim.Deconvolve(galsim_psf)
                
                image_galsim = galsim.Image(images[i,:,:,k], copy=True)
                galsim_obj = galsim.InterpolatedImage(image_galsim,scale = 0.2)
                
                deconv_galsim_obj = galsim.Convolve(inv_psf, galsim_obj)
                
                gamma1 = input_shear[0]
                gamma2 = input_shear[1]

                s = galsim.Shear(g1=gamma1,g2=gamma2)

                # Shear galsim object
                deconv_galsim_obj = deconv_galsim_obj.shear(s)

                # recconvolve by PSF
                conv_galsim_obj = galsim.Convolve(galsim_psf, deconv_galsim_obj)
                
                # Reconstruct image from sheared object
                image = galsim.ImageF(59, 59, scale=0.2)
                _ = conv_galsim_obj.drawImage(image=image)
                image_sheared[z,i,:,:,k] = image.array.data
    return image_sheared[0,i,:,:,k], image_sheared[1,i,:,:,k], image_sheared[2,i,:,:,k], image_sheared[3,i,:,:,k]


def create_model_shear(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h_psf = input_layer_2
    h = input_layer_1
    h_1, h_2, h_3, h_4 = tf.keras.layers.Lambda(lambda x: shear_img(x))(h)

    h_1 = BatchNormalization()(h_1)
    for i in range(len(filters)):
        h_psf = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_psf)
        h_psf = PReLU()(h_psf)
        h_psf = MaxPool2D()(h_psf)

        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_1)
        h_1 = PReLU()(h_1)
        h_1 = MaxPool2D()(h_1)

    h_1 = Flatten()(tf.cast(h_1,tf.float64))
    h_psf = Flatten()(tf.cast(h_psf,tf.float64))
    h_1 = tf.keras.layers.concatenate([tf.cast(h_1,tf.float64), tf.cast(h_psf,tf.float64)], axis =-1)
    h_1 = PReLU()(h_1)
    h_1 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim))(tf.cast(h_1,tf.float32))
    h_1 = tfp.layers.MultivariateNormalTriL(final_dim)(h_1)

    h_2 = BatchNormalization()(h_2)
    for i in range(len(filters)):
        h_psf = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_psf)
        h_psf = PReLU()(h_psf)
        h_psf = MaxPool2D()(h_psf)

        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = MaxPool2D()(h_2)

    h_2 = Flatten()(tf.cast(h_2,tf.float64))
    h_psf = Flatten()(tf.cast(h_psf,tf.float64))
    h_2 = tf.keras.layers.concatenate([tf.cast(h_2,tf.float64), tf.cast(h_psf,tf.float64)], axis =-1)
    h_2 = PReLU()(h_2)
    h_2 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim))(tf.cast(h_2,tf.float32))
    h_2 = tfp.layers.MultivariateNormalTriL(final_dim)(h_2)

    h_3 = BatchNormalization()(h_3)
    for i in range(len(filters)):
        h_psf = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_psf)
        h_psf = PReLU()(h_psf)
        h_psf = MaxPool2D()(h_psf)

        h_3 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_3)
        h_3 = PReLU()(h_3)
        h_3 = MaxPool2D()(h_3)

    h_3 = Flatten()(tf.cast(h_3,tf.float64))
    h_psf = Flatten()(tf.cast(h_psf,tf.float64))
    h_3 = tf.keras.layers.concatenate([tf.cast(h_3,tf.float64), tf.cast(h_psf,tf.float64)], axis =-1)
    h_3 = PReLU()(h_3)
    h_3 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim))(tf.cast(h_3,tf.float32))
    h_3 = tfp.layers.MultivariateNormalTriL(final_dim)(h_3)

    h_4 = BatchNormalization()(h_4)
    for i in range(len(filters)):
        h_psf = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_psf)
        h_psf = PReLU()(h_psf)
        h_psf = MaxPool2D()(h_psf)

        h_4 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_4)
        h_4 = PReLU()(h_4)
        h_4 = MaxPool2D()(h_4)

    h_4 = Flatten()(tf.cast(h_4,tf.float64))
    h_psf = Flatten()(tf.cast(h_psf,tf.float64))
    h_4 = tf.keras.layers.concatenate([tf.cast(h_4,tf.float64), tf.cast(h_psf,tf.float64)], axis =-1)
    h_4 = PReLU()(h_4)
    h_4 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim))(tf.cast(h_4,tf.float32))
    h_4 = tfp.layers.MultivariateNormalTriL(final_dim)(h_4)

    h_1 = PReLU()(h_1)
    h_2 = PReLU()(h_2)
    h_3 = PReLU()(h_3)
    h_4 = PReLU()(h_4)
    h = tf.keras.layers.Add([h_1,h_2,h_3,h_4])
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim))(tf.cast(h,tf.float32))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model




def create_model_wo_ls_peak_pooling_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = MaxPool2D()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model




tfd = tfp.distributions
encoded_size = 13
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)

mix = 0.3
bimix_gauss = tfd.Mixture(
  cat=tfd.Categorical(probs=[mix, 1.-mix]),
  components=[
    tfd.Normal(loc=-1., scale=0.1),
    tfd.Normal(loc=+1., scale=0.5),
])



## For image
# h = tf.keras.layers.Lambda(lambda x: tf.signal.fft3d(tf.cast(x, tf.complex64), name=None))(h)
# h_2 = tf.keras.layers.Lambda(lambda x: (tf.signal.fft3d(tf.cast(x, tf.complex64), name=None)))(h_2)
# h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
# h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft3d(tf.cast(x, tf.complex64), name=None))(h)
#### For dense layers
# h = tf.keras.layers.Lambda(lambda x: tf.signal.fft(tf.cast(x, tf.complex64), name=None))(h)
# h_2 = tf.keras.layers.Lambda(lambda x: (tf.signal.fft(tf.cast(x, tf.complex64), name=None)))(h_2)




def create_model_full_prob_flipout_dc2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    h = BatchNormalization()(input_layer_1)
    for i in range(len(filters)):
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(), 
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(), 
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h_2)
        h = PReLU()(h_2)
        #h = Dropout(0.25)(h)
    h = Flatten()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)


    model = Model([input_layer_1, input_layer_2],h)
    
    return model





# Model with coordinate of target galaxy
def create_model_wo_ls_peak_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions

    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)
    
    h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h_2)

    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
        #h = tf.keras.layers.SeparableConv2D(filters[i], (1,1), activation=None, padding='valid',  data_format='channels_last')(h)
        #h = tf.keras.layers.multiply([h,h_2]) #

    h = Flatten()(h)
    h = PReLU()(h)
    
    h = Dense(64)(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h) #Dense(2)(h)#
    #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.2))(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model







#import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.layers import util as tfp_layers_util
# Weights initialization for posteriors
def get_posterior_fn():
  return tfp_layers_util.default_mean_field_normal_fn(
      loc_initializer=tf.keras.initializers.HeNormal(), 
      untransformed_scale_initializer=tf.keras.initializers.RandomNormal(
          mean=-9, stddev=0.1)#mean=-9, stddev=0.1)
      )
# kernel divergence weight in loss
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / (100000))

def create_model_prob_flipout_peak_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    #h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(input_layer_1)
    #h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

    #h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(256, 
                                kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                kernel_divergence_fn = kernel_divergence_fn,
                                activation=None)(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model



def create_model_prob_flipout_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    #h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(input_layer_1)
    #h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        # h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same')(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        # h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same', strides=(2,2))(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        # h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        # h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same', strides=(2,2))(h)
        h = PReLU()(h)

    h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(256, 
                                kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                kernel_divergence_fn = kernel_divergence_fn,
                                activation=None)(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model



def create_model_prob_full_flipout_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    #h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(input_layer_1)
    #h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        # h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same')(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        # h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same', strides=(2,2))(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        # h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        # h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
        #                                     kernel_posterior_fn=get_posterior_fn(),
        #                                     #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
        #                                     kernel_divergence_fn=kernel_divergence_fn,
        #                                     activation=conv_activation, 
        #                                     padding='same', strides=(2,2))(h)
        h = PReLU()(h)

    h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(256, 
                                kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                kernel_divergence_fn = kernel_divergence_fn,
                                activation=None)(h)
    h = PReLU()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_prob_rt_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = tfp.layers.DenseReparameterization(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model


# tfd = tfp.distributions
encoded_size = 32
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)

def create_model_wo_ls_peak_pooling_vae(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = MaxPool2D()(h)
        #h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = Flatten()(h)
    #h_2 = Flatten()(h_2)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(1/x, tf.complex64), name=None))(h_2)

    #h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.001)
    
    
    
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = tf.keras.layers.UpSampling2D()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='sigmoid', padding='same')(h)
    #h = PReLU()(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)


    model = Model([input_layer_1, input_layer_2],h)

    return model

def create_model_wo_ls_peak_pooling_vae_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = MaxPool2D()(h)
        #h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = Flatten()(h)
    #h_2 = Flatten()(h_2)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(1/x, tf.complex64), name=None))(h_2)

    #h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.001)
    
    
    
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = tf.keras.layers.UpSampling2D()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='relu', padding='same')(h)
    #h = PReLU()(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)


    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_wo_ls_peak_pooling_vae_3(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        #h = MaxPool2D()(h)
        #h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = Flatten()(h)
    #h_2 = Flatten()(h_2)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(1/x, tf.complex64), name=None))(h_2)

    #h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    h = Flatten()(h)
    #h = PReLU()(h)
    #h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.001)
    
    
    
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32))(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = tf.keras.layers.UpSampling2D()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='sigmoid', padding='same')(h)
    #h = PReLU()(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)


    model = Model([input_layer_1, input_layer_2],h)

    return model

def create_model_wo_ls_peak_pooling_vae_4(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        #h = MaxPool2D()(h)
        #h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = Flatten()(h)
    #h_2 = Flatten()(h_2)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(1/x, tf.complex64), name=None))(h_2)

    #h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
    h = Flatten()(h)
    #h = PReLU()(h)
    #h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.001)
    
    
    
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32))(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = tf.keras.layers.UpSampling2D()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='relu', padding='same')(h)
    #h = PReLU()(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)


    model = Model([input_layer_1, input_layer_2],h)

    return model

