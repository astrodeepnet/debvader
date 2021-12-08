import os
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy
from scipy import optimize
import skimage   
from skimage import metrics
import photutils

path_folder = os.path.dirname(os.path.abspath(__file__))

#import debvader
#from debvader import model

###### TO SUPRESSS AND UNCOMMENT PREVIOUS LINES
import sys
sys.path.insert(0,'.')
import model
######

def load_deblender(survey, input_shape, latent_dim, filters, kernels, return_encoder_decoder_z = False):
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
    loading_path = str(path_folder) + "/../data/weights/" + survey + "/not_normalised/loss/"
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)


    if return_encoder_decoder_z:
        return net, encoder, decoder, z
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


def deblend_field(net, field_image, galaxy_distances_to_center, cutout_images = None, cutout_size = 59, nb_of_bands = 6, optimise_positions=False, epistemic_uncertainty_estimation=False, epistemic_criterion=100., mse_criterion=100., normalised=False):
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

    field_image = field_image.copy()

    field_size = field_image.shape[1]

    # Deblend the cutouts around the detected galaxies. If needed, create the cutouts.
    if isinstance(cutout_images, np.ndarray):
        output_images_mean, output_images_distribution = deblend(net, cutout_images, normalised=normalised)
        list_idx = list(range(0, len(output_images_mean)))
    else:
        cutout_images, list_idx = extract_cutouts(field_image, field_size, galaxy_distances_to_center, cutout_size,nb_of_bands)     
        output_images_mean, output_images_distribution = deblend(net, cutout_images[list_idx], normalised=normalised)  
    if list_idx==[]:
        print('No galaxy deblended. End of the iterative procedure.')
        return field_image, field_image, np.zeros((field_size,field_size,nb_of_bands)), np.zeros((field_size,field_size,nb_of_bands)), np.zeros((field_size,field_size,nb_of_bands)), cutout_images, output_images_mean, output_images_distribution, np.zeros((len(output_images_mean),2)), list_idx
    
    if epistemic_uncertainty_estimation:
        # Compute epistemic uncertainty (from the decoder of the deblender)
        epistemic_uncertainty = []
        for i, idx in enumerate (list_idx):
            epistemic_uncertainty.append(np.std(deblend(net, np.array([cutout_images[idx]]*100), normalised=normalised)[0], axis = 0))

    # First create padded images of the stamps at the size of the field to allow for a simple subtraction.
    output_images_mean_padded = np.zeros((len(output_images_mean),field_size,field_size,nb_of_bands))
    output_images_mean_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                          int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=output_images_mean

    # Create the corresponding standard deviation image (aleatoric uncertainty).
    output_images_stddev_padded = np.zeros((len(output_images_mean),field_size,field_size,nb_of_bands))
    output_images_stddev_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                          int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=output_images_distribution.stddev().numpy()

    # Create the corresponding epistemic uncertainty image (aleatoric uncertainty).
    output_images_epistemic_padded = np.zeros((len(output_images_mean),field_size,field_size,nb_of_bands))
    if epistemic_uncertainty_estimation:
        output_images_epistemic_padded[:,int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),
                            int((field_size-cutout_size)/2):cutout_size+int((field_size-cutout_size)/2),:]=np.array(epistemic_uncertainty)

    # Initialise a denoised field that will be composed of the deblended galaxies
    denoised_field = np.zeros((field_size,field_size,nb_of_bands))        
    denoised_field_std = np.zeros((field_size,field_size,nb_of_bands))   
    denoised_field_epistemic = np.zeros((field_size,field_size,nb_of_bands))        

    # Save an image of the field
    field_img_save = field_image.copy()

    def fun (x, img, net_output): 
        return metrics.mean_squared_error(img,scipy.ndimage.shift(net_output,shift = (x[0],x[1])))#1/skimage.measure.compare_ssim(img,scipy.ndimage.shift(net_output,shift = (x[0],x[1])))

    # Subtract each deblended galaxy to the field and add it to the denoised field.
    shifts=np.zeros((len(output_images_mean),2))
    for i,k in enumerate (list_idx):
       # Different subtraction if optimisation on positions is required
        if optimise_positions:
            opt = optimize.least_squares(fun,(0.,0.), args=(field_image[0,:,:,2],scipy.ndimage.shift(output_images_mean_padded[i,:,:,2],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))), bounds=(-3,3))
            shifts[i]=opt.x
            for j in range (nb_of_bands):
                denoised_field_std[:,:,j] +=scipy.ndimage.shift(output_images_stddev_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
                if epistemic_uncertainty_estimation:
                    denoised_field_epistemic[:,:,j] +=scipy.ndimage.shift(output_images_epistemic_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
                if ((np.sum(output_images_epistemic_padded[i,:,:,2])/np.sum(output_images_mean_padded[i,:,:,2]))>epistemic_criterion) or (metrics.mean_squared_error(cutout_images[k],output_images_mean[i])>mse_criterion): # avoid to add galaxies generated with too high uncertainty
                    pass
                else:
                    field_image[0,:,:,j] -= scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
                    denoised_field[:,:,j] +=scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0]+opt.x[0],galaxy_distances_to_center[k][1]+opt.x[1]))
        else:
            for j in range (nb_of_bands):
                denoised_field_std[:,:,j] +=scipy.ndimage.shift(output_images_stddev_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
                if epistemic_uncertainty_estimation:
                   denoised_field_epistemic[:,:,j] +=scipy.ndimage.shift(output_images_epistemic_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
                if ((np.sum(output_images_epistemic_padded[i,:,:,2])/np.sum(output_images_mean_padded[i,:,:,2]))>epistemic_criterion) or (metrics.mean_squared_error(cutout_images[k],output_images_mean[i])>mse_criterion): # avoid to add galaxies generated with too high uncertainty
                    pass
                else:
                    field_image[0,:,:,j] -= scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
                    denoised_field[:,:,j] +=scipy.ndimage.shift(output_images_mean_padded[i,:,:,j],shift = (galaxy_distances_to_center[k][0],galaxy_distances_to_center[k][1]))
 
    return field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, list_idx




def iterative_deblending(net, field_image, galaxy_distances_to_center_in, npeaks_per_iteration=10, cutout_images = None, cutout_size = 59, nb_of_bands = 6, optimise_positions=False, epistemic_uncertainty_estimation= False, epistemic_criterion=0., mse_criterion=0., normalised=False):
    '''
    Do the iterative deblending of a scene
    paramters:
        net: network used to deblend the field
        field_image: image of the field to deblend
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_images: stamps centered on the galaxies to deblend
        cutout_size: size of the stamps
        nb_of_bands: number of filters in the image
        optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
        normalised: boolean to indicate if images need to be normalised
   '''

    field_image = field_image.copy()
    
    if isinstance(galaxy_distances_to_center_in, np.ndarray):
        galaxy_distances_to_center = galaxy_distances_to_center_in
    else:    
        galaxy_distances_to_center = detect_objects(field_image, npeaks_per_iteration=npeaks_per_iteration)
    field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step = deblending_step(net, field_image, galaxy_distances_to_center, cutout_images = None, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)

    field_img_init = field_img_save.copy()
    shifts_previous = []
    k=1
    diff_mse=-1
    denoised_field_total = denoised_field
    denoised_field_std_total = denoised_field_std
    denoised_field_epistemic_total = denoised_field_epistemic
    cutout_images_total = cutout_images
    output_images_total = output_images_mean
    galaxy_distances_to_center_total = galaxy_distances_to_center
    detection_k=np.zeros((1))

    while len(detection_k)!=0:#(len(shifts)>len(shifts_previous)):

        print("iteration "+str(k))
        #print(npeaks_per_iteration)
        mse_step_previous=mse_step
        shifts_previous = shifts
        detection_k = detect_objects(field_image, npeaks_per_iteration=npeaks_per_iteration)
        # Avoid to have several detection at the same location
        idx_to_remove = []
        for i in range (len(detection_k)):
            if detection_k[i] in galaxy_distances_to_center_total:
                idx_to_remove.append(i)
        detection_k = np.delete(detection_k, idx_to_remove, axis = 0)
        detection_up_to_k = np.concatenate((detection_k,galaxy_distances_to_center_total),axis=0)
        #print(detection_k , detection_up_to_k)
        field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step = deblending_step(net, field_image, detection_k, cutout_images = None, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)

        #field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step = deblending_step(net, field_img_init, detection_up_to_k, cutout_images = None, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)
        #field_img_init=field_img_save.copy()

        denoised_field_total += denoised_field
        denoised_field_std_total += denoised_field_std
        denoised_field_epistemic_total += denoised_field_epistemic
        shifts = np.concatenate((shifts_previous, shifts), axis = 0)
        cutout_images_total = np.concatenate((cutout_images_total, cutout_images), axis = 0)
        output_images_total = np.concatenate((output_images_total, output_images_mean), axis = 0)
        galaxy_distances_to_center_total = np.concatenate((galaxy_distances_to_center_total, galaxy_distances_to_center), axis = 0)
        diff_mse = mse_step-mse_step_previous
        k+=1
        if diff_mse==0.:
            # If no galaxy is found here, except the ones that are too close from the borders, try to locate more galaxies.
            npeaks_per_iteration+=10
            detection_k = detect_objects(field_image, npeaks_per_iteration=npeaks_per_iteration)
            # Avoid to have several detection at the same location
            idx_to_remove = []
            for i in range (len(detection_k)):
                if detection_k[i] in galaxy_distances_to_center_total:
                    idx_to_remove.append(i)
            detection_k = np.delete(detection_k, idx_to_remove, axis = 0)
            if npeaks_per_iteration==50:
                print('converged on maximum peak per iteration.')
                break

        print(str(len(shifts))+' galaxies found up to this step.')
        print('deta_mse = '+str(diff_mse)+', mse_iteration = '+str(mse_step)+' and mse_previous_step = '+str(mse_step_previous))

    print('converged !')
 
    return field_img_init, field_image, denoised_field_total, denoised_field_std_total, denoised_field_epistemic_total, cutout_images_total, output_images_total


def detect_objects(field_image, npeaks_per_iteration = 10):
    '''
    Detect the objects in the field_image image using the photutils detection algorithm.
    test for dev branch
    '''
    df_temp = photutils.find_peaks(field_image[0,:,:,2], threshold=0.1, npeaks=npeaks_per_iteration, centroid_func=photutils.centroids.centroid_com)
    galaxy_distances_to_center = []

    for i in range (len(df_temp['y_peak'])):
        galaxy_distances_to_center.append((np.round(-129+df_temp['y_peak'][i]),np.round(-129+df_temp['x_peak'][i])))
            
    return np.array(galaxy_distances_to_center)


def deblending_step(net, field_image, galaxy_distances_to_center, cutout_images = None, cutout_size = 59, nb_of_bands = 6, optimise_positions=False, epistemic_uncertainty_estimation = False, epistemic_criterion=100., mse_criterion=0., normalised=False):
    '''
    One step of the iterative procedure
    paramters:
        net: network used to deblend the field
        field_image: image of the field to deblend
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_images: stamps centered on the galaxies to deblend
        cutout_size: size of the stamps
        nb_of_bands: number of filters in the image
        optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
        normalised: boolean to indicate if images need to be normalised
    '''
    
    field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, list_idx = deblend_field(net, field_image, galaxy_distances_to_center, cutout_images = cutout_images, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)
    print("Deblend "+str(len(shifts))+' more galaxy(ies)')
    for z,i in enumerate (list_idx):
        galaxy_distances_to_center[i][0] = galaxy_distances_to_center[i][0]+np.round(shifts[z][0])
        galaxy_distances_to_center[i][1] = galaxy_distances_to_center[i][1]+np.round(shifts[z][1])
    mse_step = metrics.mean_squared_error(field_img_save[0],denoised_field)

    return field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step



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
