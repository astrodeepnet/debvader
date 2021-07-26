import tensorflow as tf
import tensorflow.keras.backend as K

import pathlib
import sys  
path_folder = pathlib.Path().resolve()
sys.path.insert(0, str(path_folder)+'/tools/')
import model


def convert(my_name):
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """

    print(f"I'll convert a notebook for you some day, {my_name}.")


def load_deblender(survey, input_shape, hidden_dim, latent_dim, filters, kernels):
    """
    load weights trained for a particular dataset
    parameters:
        survey: string calling the particular dataset (choices are: "dc2")
        input_shape: shape of input tensor
        hidden_dim: size of the two dense layers before and after the latent space
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """
    # Create the model
    net = model.create_model_vae(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation=None, dense_activation=None)

    # Define the loss function
    def vae_loss(x, x_decoded_mean):
        xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
        return xent_loss

    # Compile the model
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              loss=vae_loss,
              experimental_run_tf_function=False)

    # Load the weights corresponding to the chosen survey
    loading_path = str(path_folder)+'/weights/'+survey+'/'
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    return net