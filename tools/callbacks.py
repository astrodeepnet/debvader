import sys, os
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization, Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import metrics



###### Callbacks
# Create a callback for changing KL coefficient in the loss

class changeAlpha(Callback):
    def __init__(self, alpha, network, loss, metric):
        self.epoch = 1
        self.alpha = alpha
        self.network = network
        self.loss = loss
        self.metric = metric
        #self.path = path
        #self.epochs = epochs
    def on_epoch_end(self, alpha, network):
        stable = 1
        #new_alpha = 0.3
        if self.epoch > stable and K.get_value(self.alpha)<1 :
            #if (self.alpha < 1):
                #new_alpha =1
            #else:
            print('Changing loss')
            new_alpha = K.get_value(self.alpha)+0.05
            if new_alpha > 1:
                new_alpha = 1
            print(new_alpha, self.epoch)
            K.set_value(self.alpha, new_alpha)
            #self.loss = self.loss(alpha)
            negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)+ sum(self.network.losses) *(K.get_value(self.alpha)-1)
            self.network.compile('adam', loss=negative_log_likelihood, metrics=['mse', 'acc', self.metric])
            K.set_value(self.network.optimizer.lr, 0.0001)
            print('loss modified')
            self.epoch = 1
            #np.save(self.path+'alpha', K.get_value(self.alpha))
        
        self.epoch +=1

# Create a callback to change learning rate of the optimizer during training
class changelr(Callback):
    def __init__(self,vae):
        self.epoch = 0
        self.vae = vae
    
    def on_epoch_end(self, alpha, vae):
        if (self.epoch == 100):
            self.epoch =0
            # actual_value = K.get_value(self.vae.optimizer.lr)
            # if (actual_value > 0.000009):
            #     new_value = actual_value/2
            #     K.set_value(self.vae.optimizer.lr, new_value)
            #     print(K.get_value(self.vae.optimizer.lr))
            if self.epoch == 200:
                K.set_value(self.vae.optimizer.lr, 10-5)
        self.epoch +=1


# # Create a callback to monitor training
# class wandb_cb(Callback):
#     def __init__(self,vae):
#         self.epoch = 0
#         self.vae = vae
    
#     def on_epoch_end(self, alpha, vae):
#         wandb.log({"Epoch": epoch,        
#            "Train Loss": loss_train,        
#            "Train Acc": acc_train,        
#            "Valid Loss": loss_valid,        
#            "Valid Acc": acc_valid})
#         self.epoch +=1

