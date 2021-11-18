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
        net: neural network used to do the deblending
        images: array of images. It can contain only one image.
    """
    # Normalize the images
    #images_normed = np.tanh(np.arcsinh(images))

    # Denorm deblended images
    #images_deblended = np.sinh(np.arctanh(net.predict(images_normed)))

    return net(tf.cast(images, tf.float32)).mean().numpy(), net(tf.cast(images, tf.float32))


def deblend_field(net, field_image, galaxy_distances_to_center, cutout_images = None, cutout_size = 59, nb_of_bands = 6):
    """
    Deblend a field of galaxies
    parameters:
        net: network used to deblend the field
        field_image: image of the field to deblend
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_images: stamps centered on the galaxies to deblend
        cutout_size: size of the stamps
    """
    field_size = field_image.shape[1]

    # Deblend the cutouts around the detected galaxies. If needed, create the cutouts.
    if cutout_images!= None:
        output_images_mean, output_images_distribution = deblend(net, cutout_images)
    else:
        cutout_images = extract_cutouts(field_image, field_size, galaxy_distances_to_center, cutout_size,nb_of_bands)        
        output_images_mean, output_images_distribution = deblend(net, cutout_images)
    
    # First create padded images of the stamps at the size of the field to allow for a simple subtraction.
    output_images_mean_padded = np.zeros((len(cutout_images),field_size,field_size,nb_of_bands))
    output_images_mean_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                          int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=output_images_mean

    # Initialise a denoised field that will be composed of the deblended galaxies
    denoised_field = np.zeros((field_size,field_size,nb_of_bands))        

    # Save an image of the field
    field_img_save = field_image.copy()
    
    # Subtract each deblended galaxy to the field and add it to the denoised field.
    for i in range(len(output_images_mean)):
        field_image[0] -= np.roll(output_images_mean_padded[i], (galaxy_distances_to_center[i][0],galaxy_distances_to_center[i][1]), axis = (0,1))
        denoised_field +=np.roll(output_images_mean_padded[i], (galaxy_distances_to_center[i][0],galaxy_distances_to_center[i][1]), axis = (0,1))   

    return field_img_save, field_image, denoised_field, cutout_images, output_images_mean



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
    for i in range(len(galaxy_distances_to_center)):
        try:
            x_shift = galaxy_distances_to_center[i][0]
            y_shift = galaxy_distances_to_center[i][1]
            cutout_images[i]=field_image[0,-int(cutout_size/2)+x_shift+int(field_size/2):int(cutout_size/2)+x_shift+int(field_size/2)+1,
                                    -int(cutout_size/2)+y_shift+int(field_size/2):int(cutout_size/2)+y_shift+int(field_size/2)+1]
        except:
            print("Galaxy "+str(i)+" is too close from the border of the field to be considered here.")

    return cutout_images