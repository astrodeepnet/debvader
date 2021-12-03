import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy
from scipy import optimize
import skimage

path_folder = os.path.dirname(os.path.abspath(__file__))

import debvader
from debvader import model


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
    net, encoder, decoder = model.create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
    )

    # Define the loss function
    def vae_loss(x, x_decoded_mean):
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
    loading_path = str(path_folder) + "/../data/weights/" + survey + "/deblender/val_loss/"
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)


    if return_encoder_decoder:
        return net, encoder, decoder
    else:
        return net



def deblend(net, images, normalised = False):
    """
    Deblend the image using the network
    parameters:
        net: neural network used to do the deblending
        images: array of images. It can contain only one image
        normalised: boolean to indicate if images need to be normalised
    """
    if normalised:
        # Normalize input images
        images_normed = np.tanh(np.arcsinh(images))
        # Denorm output images
        images = np.sinh(np.arctanh(net.predict(images_normed)))

    return net(tf.cast(images, tf.float32)).mean().numpy(), net(tf.cast(images, tf.float32))



def deblend_field(net, field_image, galaxy_distances_to_center, cutout_images = None, cutout_size = 59, nb_of_bands = 6, optimise_positions=False, normalised=False):
    """
    Deblend a field of galaxies
    parameters:
        net: network used to deblend the field
        field_image: image of the field to deblend
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_images: stamps centered on the galaxies to deblend
        cutout_size: size of the stamps
        nb_of_bands: number of filters in the image
        optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
        normalised: boolean to indicate if images need to be normalised
    """
    field_size = field_image.shape[1]

    # Deblend the cutouts around the detected galaxies. If needed, create the cutouts.
    if isinstance(cutout_images, np.ndarray):
        output_images_mean, output_images_distribution = deblend(net, cutout_images, normalised=normalised)
        list_idx = list(range(0, len(output_images_mean)))
    else:
        cutout_images, list_idx = extract_cutouts(field_image, field_size, galaxy_distances_to_center, cutout_size,nb_of_bands)     
        output_images_mean, output_images_distribution = deblend(net, cutout_images[list_idx], normalised=normalised)  

    # First create padded images of the stamps at the size of the field to allow for a simple subtraction.
    output_images_mean_padded = np.zeros((len(output_images_mean),field_size,field_size,nb_of_bands))
    output_images_mean_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                          int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=output_images_mean

    # Create the corresponding standard deviation image (aleatoric uncertainty).
    output_images_stddev_padded = np.zeros((len(output_images_mean),field_size,field_size,nb_of_bands))
    output_images_stddev_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                          int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=output_images_distribution.stddev().numpy()

    # Initialise a denoised field that will be composed of the deblended galaxies
    denoised_field = np.zeros((field_size,field_size,nb_of_bands))        
    denoised_field_std = np.zeros((field_size,field_size,nb_of_bands))          

    # Save an image of the field
    field_img_save = field_image.copy()

    def fun (x, img, net_output): 
        return skimage.measure.compare_mse(img,scipy.ndimage.shift(net_output,shift = (x[0],x[1])))

    # Subtract each deblended galaxy to the field and add it to the denoised field.
    shifts=np.zeros((len(output_images_mean),2))
    for i,k in enumerate (list_idx):
       # Different subtraction if optimisation on positions is required
        if optimise_positions:
            opt = optimize.least_squares(fun,(0.,0.), args=(field_image[0,:,:,2],scipy.ndimage.shift(output_images_mean_padded[i,:,:,2],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))), bounds=(-3,3))
            shifts[i]=opt.x
            for j in range (nb_of_bands):
                denoised_field_std[:,:,j] +=scipy.ndimage.shift(output_images_stddev_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
                field_image[0,:,:,j] -= scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
                denoised_field[:,:,j] +=scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
        else:
            for j in range (nb_of_bands):
                denoised_field_std[:,:,j] +=scipy.ndimage.shift(output_images_stddev_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
                field_image[0,:,:,j] -= scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
                denoised_field[:,:,j] +=scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
 
    return field_img_save, field_image, denoised_field, denoised_field_std, cutout_images, output_images_mean, output_images_distribution, shifts, list_idx





def extract_cutouts(field_image, field_size, galaxy_distances_to_center,cutout_size=59, nb_of_bands = 6):
    """
    Extract the cutouts around particular galaxies in the field
    parameters:
        field_image: image of the field to deblend
        field_size: size of the field
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_size: size of the stamps
    """
    cutout_images = np.zeros((len(galaxy_distances_to_center),cutout_size, cutout_size, nb_of_bands))
    list_idx = []
    flag = False
    for i in range(len(galaxy_distances_to_center)):
        try:
            x_shift = galaxy_distances_to_center[i][0]
            y_shift = galaxy_distances_to_center[i][1]
            cutout_images[i]=field_image[0,-int(cutout_size/2)+x_shift+int(field_size/2):int(cutout_size/2)+x_shift+int(field_size/2)+1,
                                    -int(cutout_size/2)+y_shift+int(field_size/2):int(cutout_size/2)+y_shift+int(field_size/2)+1]
            list_idx.append(i)
        except:
            flag = True

    if flag:
        print("Some galaxies are too close from the border of the field to be considered here.")
            
    return cutout_images, list_idx
