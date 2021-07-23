import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import galsim
import random
import cmath as cm
import math
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, BatchNormalization, Reshape, Flatten, Conv2D,  PReLU,Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from scipy.stats import norm
import tensorflow as tf

from . import model, vae_functions, plot, layers

I_lsst = np.array([255.2383, 2048.9297, 3616.1757, 4441.0576, 4432.7823, 2864.145])
I_euclid = np.array([5925.8097, 3883.7892, 1974.2465,  413.3895])
beta = 2.5


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

############# Normalize data ############# 
def norm(x, bands,test_dir, channel_last=False, inplace=True):
    I = np.load(test_dir+'galaxies_blended_20191024_0_I_norm.npy', mmap_mode = 'c')
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.tanh(np.arcsinh(y[i,:,:,ib]/(I[b]/beta)))
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.tanh(np.arcsinh(y[i,ib]/(I[b]/beta)))
    return y

def denorm(x, bands,test_dir, channel_last=False, inplace=True):
    I = np.load(test_dir+'galaxies_blended_20191024_0_I_norm.npy', mmap_mode = 'c')
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.sinh(np.arctanh(y[i,:,:,ib]))*(I[b]/beta)
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.sinh(np.arctanh(x[i,ib]))*(I[b]/beta)
    return y

# Here we do the detection in R band of LSST
def SNR_peak(gal_noiseless, sky_background_pixel, band=6, snr_min=2):
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    snr = np.max(gal_noiseless[band])/sky_background_pixel[band]
    return (snr>snr_min), snr


def SNR(gal_noiseless, sky_background_pixel, band=6, snr_min=5):
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    signal = gal_noiseless[band]
    variance = signal+sky_background_pixel[band] # for a Poisson process, variance=mean
    snr = np.sqrt(np.sum(signal**2/variance))
    return (snr>snr_min), snr


############# COMPUTE BLENDEDNESS #############
def compute_blendedness_single(image1, image2):
    """
    Return blendedness computed with two images of single galaxy created with GalSim

    Parameters
    ----------
    img, img_new : GalSim images convolved with its PSF and drawn in its filter
    """
    if isinstance(image1, galsim.image.Image):
        im1 = np.array(image1.array.data)
        im2 = np.array(image2.array.data)
    else:
        im1 = image1
        im2 = image2
    # print(image,image_new)
    blnd = np.sum(im1*im2)/np.sqrt(np.sum(im1**2)*np.sum(im2**2))
    return blnd

def compute_blendedness_total(img_central, img_others):
    if isinstance(img_central, galsim.image.Image):
        ic = np.array(img_central.array.data)
        io = np.array(img_others.array.data)
    else :
        ic = img_central
        io = img_others
    itot = ic + io
    # print(image,image_new)
    # blnd = np.sum(ic*io)/np.sum(io**2)
    # blnd = 1. - compute_blendedness_single(itot,io)
    blnd = 1. - np.sum(ic*ic)/np.sum(itot*ic)
    return blnd

def compute_blendedness_aperture(img_central, img_others, radius):
    if isinstance(img_central, galsim.image.Image):
        ic = np.array(img_central.array.data)
        io = np.array(img_others.array.data)
    else :
        ic = img_central
        io = img_others
    h, w = ic.shape
    mask = plot.createCircularMask(h, w, center=None, radius=radius)
    flux_central = np.sum(ic*mask.astype(float))
    flux_others = np.sum(io*mask.astype(float))
    return flux_others / (flux_central+flux_others)

############ DELTA_R and DELTA_MAG COMPUTATION FOR MOST BLENDED GALAXY WITH THE CENTERED ONE ##########
def compute_deltas_for_most_blended(shift,mag,blendedness):#(shift_path, mag_path):
    #mag =np.load(mag_path)
    #shift =np.load(shift_path)

    # Create an array of minimum magnitude and maximum blendedness for each image
    mag_min = np.zeros(len(mag))
    blend_max = np.zeros(len(blendedness))
    for k in range (len(mag)):
        mag_min[k] = np.min(mag[k])
        if (len(blendedness[k])>=1):
            blend_max[k] = np.max(blendedness[k])
        else:
            blend_max[k] = 0

    # set lists
    deltas_r= np.zeros((len(shift),3))
    delta_r= np.zeros((len(shift)))
    delta_mag = np.zeros((len(shift)))
    deltas_mag= np.zeros((len(shift),4))

    for i in range (len(shift)):
        for j in range (len(shift[i])):
            deltas_r[i][j] = np.sqrt(np.square(shift[i][j][0])+np.square(shift[i][j][1]))
        for j in range (len(mag[i])):
            deltas_mag [i][j] = mag[i][j] - mag_min[i]
            
    # Create a deltas_mag liste without all zeros: place of the centered galaxy when generated
    deltas_mag_3= np.zeros((len(deltas_mag),3))
    counter = 0
    for k in range (len(deltas_mag)):
        No_zero = True
        for l in range (len(deltas_mag[k])):
            if deltas_mag[k][l] == 0 and No_zero:
                counter +=1
                No_zero = False
            elif No_zero == False:
                deltas_mag_3[k][l-1] = deltas_mag[k][l]
            else:
                deltas_mag_3[k][l] = deltas_mag[k][l]
    
    # Return delta_mag and delta_r for most blended galaxies
    for i in range (len(blendedness)):
        for k in range (len(blendedness[i])):
            if blendedness[i][k] == blend_max[i]:
                delta_mag[i] = deltas_mag_3[i,k]
                delta_r[i ]=  deltas_r[i,k]

    return delta_r, delta_mag, blend_max

############ DELTA_R and DELTA_MAG COMPUTATION FOR DELTA_R MIN ##########
def delta_min(shift,mag):#(shift_path, mag_path):
    #mag =np.load(mag_path)
    #shift =np.load(shift_path)

    # Create an array of minimum magnitude for each image
    mag_min = np.zeros(len(mag))
    for k in range (len(mag)):
        mag_min[k] = np.min(mag[k])

    # set lists
    deltas_r= np.zeros((len(shift),3))
    delta_r= np.zeros((len(shift)))
    delta_mag = np.zeros((len(shift)))
    deltas_mag= np.zeros((len(shift),4))

    for i in range (len(shift)):
        for j in range (len(shift[i])):
            deltas_r[i][j] = np.sqrt(np.square(shift[i][j][0])+np.square(shift[i][j][1]))
        for j in range (len(mag[i])):
            deltas_mag [i][j] = mag[i][j] - mag_min[i]
            
    # Create a deltas_mag liste without all zeros: place of the centered galaxy when generated
    deltas_mag_3= np.zeros((len(deltas_mag),3))
    counter = 0
    for k in range (len(deltas_mag)):
        No_zero = True
        for l in range (len(deltas_mag[k])):
            if deltas_mag[k][l] == 0 and No_zero:
                counter +=1
                No_zero = False
            elif No_zero == False:
                deltas_mag_3[k][l-1] = deltas_mag[k][l]
            else:
                deltas_mag_3[k][l] = deltas_mag[k][l]
                    
    # Take the min of the non zero delta r
    c = 0
    for j in range (len(shift)):
        # If all the deta_r are equals to 0 (there is only on galaxy on the image) then write 0
        if (deltas_r[j,:].any() == 0):
            delta_r[j] = 0
            delta_mag[j] = 0
            c+=1
        else:
            x = np.where(deltas_r[j] == 0)[0]
            deltas = np.delete(deltas_r[j],x)
            delta_r[j] = np.min(deltas)
            y = np.where(deltas == np.min(deltas))[0]
            delta_mag[j] = deltas_mag_3[j,y]
        
    return delta_r, delta_mag

############# LOAD MODEL ##################
def load_vae_conv(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    #### Build the model
    vae_loaded, Dkl, z = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, encoder, decoder, z

def load_model(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    #### Build the model
    loaded_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

    if folder == False: 
        loaded_model.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        loaded_model.load_weights(latest)

    return loaded_model

def load_vae_full(path, nb_of_bands, folder=False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model_vae.vae_model(latent_dim, nb_of_bands)
    #z, Dkl = layers.SampleMultivariateGaussian(full_cov=False, add_KL=False, return_KL=True, coeff_KL=0)(encoder.outputs)

    #### Build the model
    vae_loaded, Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)
    
    #encoder.trainable = False

    #net = Model(inputs=encoder.inputs, outputs=decoder(z))
    return  vae_loaded, encoder#net

def load_alpha(path_alpha):
    return np.load(path_alpha+'alpha.npy')



##############   LOAD DATA    ############
def delta_r_min(shift_path):
    """
    Function to compute the delta_r from the shift saved
    """
    shift =np.load(shift_path)
    
    # set lists
    deltas_r= np.zeros((len(shift),4))
    delta_r= np.zeros((len(shift)))
    
    # compute the delta r for each couple of galaxies
    for i in range (4):
        deltas_r[:,i] = np.sqrt(np.square(shift[:,i,0])+np.square(shift[:,i,1]))

    # Take the min of the non zero delta r
    for j in range (len(shift)):
        if (deltas_r[j,:].any() == 0):
            delta_r[j] = 0
        else:
            x = np.where(deltas_r[j] == 0)[0]
            deltas = np.delete(deltas_r[j],x)
            delta_r[j] = np.min(deltas)
    
    return delta_r


##############   MULTIPROCESSING    ############
import multiprocessing
import time
from tqdm import tqdm, trange

def apply_ntimes(func, n, args, verbose=True, timeout=None):
    """
    Applies `n` times the function `func` on `args` (useful if, eg, `func` is partly random).
    Parameters
    ----------
    func : function
        func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
    n : int
    args : any
    timeout : int or float
        If given, the computation is cancelled if it hasn't returned a result before `timeout` seconds.
    Returns
    -------
    type
        Result of the computation of func(iter).
    """
    pool = multiprocessing.Pool()

    multiple_results = [pool.apply_async(func, args) for _ in range(n)]

    pool.close()
    
    return [res.get(timeout) for res in tqdm(multiple_results, desc='# castor.parallel.apply_ntimes', disable = True)]
