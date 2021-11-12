import os

import tensorflow as tf


def train_network(net, epochs, training_data, validation_data, batch_size, callbacks, verbose):
    """
    train a network on data for a fixed number of epochs
    parameters:
        net: network to train
        epochs: number of epochs
        training_data: training data under the format of numpy arrays (inputs, labels)
        validation_data: validation data under the format of numpy arrays (inputs, labels)
        batch_size: size of batch for training
        callbacks: callbacks wanted for the training
        verbose: display of training (1:yes, 2: no)
    """

    hist = net.fit(training_data[0], training_data[1], epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                    shuffle=True,
                    validation_data=(validation_data[0], validation_data[1]),
                    validation_steps=int(len(validation_data[0])/batch_size),
                    callbacks=callbacks)

    return hist

def train_deblender(survey_name, epochs, training_data, validation_data, batch_size, callbacks, verbose):
    """
    function to train a network for a new survey
    survey_name: name of the survey
    epochs: number of epochs of training
    training_data: training data under the format of numpy arrays (inputs, labels)
    validation_data: validation data under the format of numpy arrays (inputs, labels)
    batch_size: size of batch for training
    callbacks: callbacks wanted for the training
    verbose: display of training (1:yes, 2: no)
   
    """
    # Generate a network for training. The architecture is fixed.
    net, encoder, decoder = create_model_vae(input_shape, latent_dim, filters, kernels, conv_activation=None, dense_activation=None)
    net.summary()

    # Define the loss as the log likelihood of the distribution on the image pixels
    def vae_loss(x, x_decoded_mean):
        return -x_decoded_mean.log_prob(x)

    # Custom metric to display the KL divergence during training
    def kl_metric(y_true, y_pred):
        return K.sum(net.losses)

    # Compilation
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
                loss=vae_loss,
                metrics = ['mse', kl_metric],
                experimental_run_tf_function=False,
                )

    if transfer_learning:
        # Start from the weights of an already trained network (recommended)
        path_output = '../data/weigths/dc2/not_normalised/'
        latest = tf.train.latest_checkpoint(path_output)
        net.load_weights(latest)

    # Callbacks
    saving_path = '../data/weigths/'+str(survey_name)+'/'
    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'val_mse/weights_noisy_v4.{epoch:02d}-{val_mse:.2f}.ckpt', monitor='val_mse', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'val_loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'mse/weights_noisy_v4.{epoch:02d}-{mse:.2f}.ckpt', monitor='mse', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'loss/weights_noisy_v4.{epoch:02d}-{loss:.2f}.ckpt', monitor='loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

    callbacks = [checkpointer_mse, checkpointer_loss, checkpointer_val_mse, checkpointer_val_loss]

    # Do the training for the VAE
    hist = training(net, epochs, training_data, validation_data, batch_size, callbacks, verbose)

    return hist, net


def train_network_custom(net, loss, epochs, training_data, validation_data, batch_size, callbacks, verbose):
    
    # Show architecture of the network to train
    net.summary()

    # Custom metric to display the KL divergence during training
    def kl_metric(y_true, y_pred):
        return K.sum(net.losses)

    # Compilation
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
                loss=vae_loss,
                metrics = ['mse', kl_metric],
                experimental_run_tf_function=False,
                )

    # 
    path_output = '/pbs/home/b/barcelin/sps_link/TFP/weights/test_vae_uncertainty/'+str(sys.argv[1])+'/'

    latest = tf.train.latest_checkpoint(path_output)
    net.load_weights(latest)

    # Callbacks
    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'val_mse/weights_noisy_v4.{epoch:02d}-{val_mse:.2f}.ckpt', monitor='val_mse', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'val_loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'mse/weights_noisy_v4.{epoch:02d}-{mse:.2f}.ckpt', monitor='mse', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
    checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_output+'loss/weights_noisy_v4.{epoch:02d}-{loss:.2f}.ckpt', monitor='loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

    callbacks = [checkpointer_mse, checkpointer_loss, checkpointer_val_mse, checkpointer_val_loss]
