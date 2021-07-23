from deblender.tools import model, 

def convert(my_name):
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """

    print(f"I'll convert a notebook for you some day, {my_name}.")


def load_deblender(survey):
    """
    load weights trained for a particular dataset
    parameters:
        survey: string calling the particular dataset
    """
    net = model.create_model_wo_ls_peak_pooling_vae_4(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

    def vae_loss(x, x_decoded_mean):
        xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
        #x_res = K.mean(K.sum(K.binary_crossentropy(x, x-x_decoded_mean), axis=[1,2,3]))
        return xent_loss

    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              loss=vae_loss,
              experimental_run_tf_function=False)

    loading_path = '../weights/'+survey+'/'
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    return net