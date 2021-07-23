# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn import preprocessing
from random import choice

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils
import tensorflow as tf
import galsim

from skimage import restoration
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve




## Compute lensed ellipticities from shear and 
def calc_lensed_ellipticity_1(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =  es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.real(e)

def calc_lensed_ellipticity_2(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =   es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.imag(e)

def calc_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =   es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.absolute(e)





class BatchGenerator_dc2_deconv_noisy_2(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_1')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_1')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]
        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_cropped_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_cropped_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_cropped_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))


        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]
        

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y



class BatchGenerator_redshift(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_cropped_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_cropped_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_cropped_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))


        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 1))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]
        #redshift = new_data['redshift'][indices]
        y[:,0]= new_data['redshift'][indices]
        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            #y[:,0] = -ellipticity_1
            #y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            #y[:,0] = ellipticity_1
            #y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            #y[:,0] = ellipticity_1
            #y[:,1] = -ellipticity_2
        #else:
            #y[:,0] = -ellipticity_1
            #y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y



class BatchGenerator_redshift_ellipticity(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_cropped_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_cropped_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_cropped_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))


        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 3))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]
        #redshift = new_data['redshift'][indices]
        y[:,2]= new_data['redshift'][indices]
        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y



class BatchGenerator_dc2_deconvolution(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_deconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_deconvolved_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_deconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_deconvolved_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_deconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_deconvolved_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_deconvolved_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_deconvolved_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_deconvolved_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_deconvolved_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_deconvolved_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))


        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y




class BatchGenerator_dc2_reconvolution(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_reconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_reconvolved_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_reconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_reconvolved_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_reconvolved_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_reconvolved_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_reconvolved_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_reconvolved_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_reconvolved_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_reconvolved_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_reconvolved_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))


        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y



class BatchGenerator_dc2_deconv_noisy_2_deconvolution(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        #if self.epoch == self.step_size:
        #    print('end of epoch')
        #    saving_path = self.saving_path+'/end_step/'
        #    self.net.save_weights(saving_path+'cp-'+str(self.epoch)+'.ckpt')
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            #print('training')
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]
        
        if self.trainval_or_test == 'validation':
            #print('validation')
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_cropped_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_cropped_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_cropped_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))

        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, -shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, -shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, -shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        for i in range(len(x_1[:,0,0,0])):
            for j in range(len(x_1[0,0,0])):
                x_1[i,:,:,j], _  = restoration.deconvolution.unsupervised_wiener(x_1[i,:,:,j], x_2[i,:,:,j])

        y = np.zeros((self.batch_size, 2))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        
        return (x_1, x_2), y



class BatchGenerator_dc2_redshift(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        #print(sample_filename)
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('sample.npy','data.csv'))
        psf = np.load(sample_filename.replace('img_sample.npy','psf_sample.npy'), mmap_mode = 'c')

        new_data = data[(np.abs(data['e1'])<=1.) &
                        (np.abs(data['e2'])<=1.)]
            
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = sample[indices][:,:,:,self.bands]
        x_2 = psf[indices][:,:,:,self.bands]
        #print(x_1.shape)
        #print(x_2.shape)

        y = new_data['redshift'][indices]

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)

        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)
        
        y = np.expand_dims(y, axis=-1)

        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return (x_1, x_2), y
        elif self.trainval_or_test == 'test':
            return (x_1, x_2), y




class BatchGenerator_dc2_deconv_vae(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        #index = np.random.choice(list(range(len(self.p))), p=self.p)
        #sample_filename = self.list_of_samples[index]

        sample = np.load(sample_filename, mmap_mode = 'c')
        sample_dc2 = np.load(sample_filename.replace('img_noiseless_sample','img_noiseless_sample'), mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
        psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
        #print(len(data['e1']))
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))
        new_data = data[(np.abs(data['e1'])<=1.) &
                        (np.abs(data['e2'])<=1.)]# &
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        #print(len(new_data['e1']))
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        ell_1 = np.array(new_data['e1'][indices])
        ell_2 = np.array(new_data['e2'][indices])
        shear_1 = np.array(new_data['shear_1'][indices])
        shear_2 = np.array(new_data['shear_2'][indices])
        convergence = np.array(new_data['convergence'][indices])
        
        ellipticity = calc_lensed_ellipticity(ell_1, ell_2, shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(ell_1, ell_2, shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(ell_1, ell_2, shear_1, shear_2, convergence))

        x_1 = np.tanh(np.arcsinh(sample_dc2[indices][:,:,:,self.bands]))
        x_3 = np.tanh(np.arcsinh(sample[indices][:,:,:,self.bands]))
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            x_3 = np.flip(x_3, axis=2)
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1 # sign changed
            #y[:,1] = -quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            x_3 = np.swapaxes(x_3, 2, 1)
            #y[:,2] = new_data['redshift'][indices]         
            #y[:,0] = -quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = +ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            x_3 = np.swapaxes(np.flip(x_3, axis=2), 2, 1)
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = -quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = -quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = +ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)
            x_3 = np.expand_dims(x_3, axis=-1)

        #print(y)
        #x_1 = np.arcsinh(x_1/0.01)
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return (x_1, x_2), x_3
        elif self.trainval_or_test == 'test':
            return (x_1, x_2), x_3



class BatchGenerator_dc2_deconv_vae_noisy(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        #index = np.random.choice(list(range(len(self.p))), p=self.p)
        #sample_filename = self.list_of_samples[index]

        sample = np.load(sample_filename, mmap_mode = 'c')
        sample_dc2 = np.load(sample_filename.replace('img_noiseless_sample','img_cropped_sample'), mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
        psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
        #print(len(data['e1']))
        data['weights']=(np.abs(data['e1'])+np.abs(data['e2']))
        new_data = data[(np.abs(data['e1'])<=1.) &
                        (np.abs(data['e2'])<=1.)]# &
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        #print(len(new_data['e1']))
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = None)#new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        ell_1 = np.array(new_data['e1'][indices])
        ell_2 = np.array(new_data['e2'][indices])
        shear_1 = np.array(new_data['shear_1'][indices])
        shear_2 = np.array(new_data['shear_2'][indices])
        convergence = np.array(new_data['convergence'][indices])
        
        ellipticity = calc_lensed_ellipticity(ell_1, ell_2, shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(ell_1, ell_2, shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(ell_1, ell_2, shear_1, shear_2, convergence))

        x_1 = np.tanh(np.arcsinh(sample_dc2[indices][:,:,:,self.bands]))
        x_3 = np.tanh(np.arcsinh(sample[indices][:,:,:,self.bands]))
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            x_3 = np.flip(x_3, axis=2)
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1 # sign changed
            #y[:,1] = -quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            x_3 = np.swapaxes(x_3, 2, 1)
            #y[:,2] = new_data['redshift'][indices]         
            #y[:,0] = -quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = +ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            x_3 = np.swapaxes(np.flip(x_3, axis=2), 2, 1)
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = -quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = -quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = +ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            #y[:,2] = new_data['redshift'][indices]
            #y[:,0] = quantile_transformer_1.transform(ellipticity_1.reshape(-1, 1))[:,0]#ellipticity_1
            #y[:,1] = quantile_transformer_2.transform(ellipticity_2.reshape(-1, 1))[:,0]#ellipticity_2
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)
            x_3 = np.expand_dims(x_3, axis=-1)

        #print(y)
        #x_1 = np.arcsinh(x_1/0.01)
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return (x_1, x_2), x_3
        elif self.trainval_or_test == 'test':
            return (x_1, x_2), x_3

