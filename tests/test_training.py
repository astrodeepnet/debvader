# Load librairies
import os
import numpy as np
import tensorflow as tf
import pkg_resources

from debvader import train
# from debvader import normalization


def test_training():
    # Load data
    data_folder_path = pkg_resources.resource_filename('debvader', "data/")
    image_path = os.path.join(data_folder_path + '/dc2_imgs/imgs_dc2.npy')
    images = np.load(image_path, mmap_mode = 'c')

    # Divide dataset into training and validation datasets
    training_data_vae = np.array((images[:5], images[:5]))
    validation_data_vae = np.array((images[5:], images[5:]))

    training_data_deblender = np.array((images[:5], images[:5]))
    validation_data_deblender = np.array((images[5:], images[5:]))


    # Test training
    hist_vae, hist_deblender, net = train.train_deblender("lsst",
                                                        from_survey = "dc2", 
                                                        epochs = 2, 
                                                        training_data_vae = training_data_vae, 
                                                        validation_data_vae = validation_data_vae, 
                                                        training_data_deblender = training_data_deblender, 
                                                        validation_data_deblender = validation_data_deblender)
