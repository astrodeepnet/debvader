import os

import pkg_resources
import tensorflow as tf
import tensorflow.keras.backend as K

from debvader import model


def train_network(
    net, epochs, training_data, validation_data, batch_size, callbacks, verbose=1
):
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

    print("\nStart the training")
    hist = net.fit(
        training_data[0],
        training_data[1],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
        validation_data=(validation_data[0], validation_data[1]),
        validation_steps=int(len(validation_data[0]) / batch_size),
        callbacks=callbacks,
    )

    return hist


def define_callbacks(vae_or_deblender, survey_name):
    """
    Define callbacks for a network to train
    parameters:
        vae_or_deblender: training a VAE or a deblender. Used for the saving path.
        survey_name: name of the survey from which the data comes. Used for the saving path.
    """
    data_path = pkg_resources.resource_filename("debvader", "data/")

    saving_path = os.path.join(
        data_path, "weights/", str(survey_name), str(vae_or_deblender), ""
    )
    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(
        filepath=saving_path + "val_mse/weights_noisy_v4.ckpt",
        monitor="val_mse",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=saving_path + "val_loss/weights_noisy_v4.ckpt",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    callbacks = [checkpointer_val_mse, checkpointer_val_loss]

    return callbacks


def train_deblender(
    survey_name,
    from_survey,
    epochs,
    training_data_vae,
    validation_data_vae,
    training_data_deblender,
    validation_data_deblender,
    nb_of_bands=6,
    channel_last=True,
    batch_size=5,
    verbose=2,
):
    """
    function to train a network for a new survey
    survey_name: name of the survey
    from_survey: name of the survey used for transfer learning. The weights saved for this survey will be loaded as initialisation of the network.
    epochs: number of epochs of training
    training_data_{}: training data under the format of numpy arrays (inputs, labels) for the vae or the deblender
    validation_data_{}: validation data under the format of numpy arrays (inputs, labels) for the vae or the deblender
    batch_size: size of batch for training
    callbacks: callbacks wanted for the training
    verbose: display of training (1:yes, 2: no)
    """
    # Generate a network for training. The architecture is fixed.
    input_shape = (59, 59, nb_of_bands)
    latent_dim = 32
    filters = [32, 64, 128, 256]
    kernels = [3, 3, 3, 3]

    net, encoder, decoder, z = model.create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
        conv_activation=None,
        dense_activation=None,
    )
    print("VAE model")
    net.summary()

    # Define the loss as the log likelihood of the distribution on the image pixels
    def vae_loss(x, x_decoded_mean):
        return -x_decoded_mean.log_prob(x)

    # Custom metric to display the KL divergence during training
    def kl_metric(y_true, y_pred):
        return K.sum(net.losses)

    # Compilation
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        metrics=["mse", kl_metric],
        experimental_run_tf_function=False,
    )

    # Check if data format is correct
    if not channel_last & (training_data_vae.shape[2] != nb_of_bands):
        print(
            "The number of bands in the data does not correspond to the number of filters in the network. Correct this before starting again."
        )
        raise ValueError
    if channel_last & (training_data_vae.shape[-1] != nb_of_bands):
        print(
            "The number of bands in the data does not correspond to the number of filters in the network. Correct this before starting again."
        )
        raise ValueError

    # Start from the weights of an already trained network (recommended if possible)
    if from_survey is not None:

        data_path = pkg_resources.resource_filename("debvader", "data/")
        path_output = os.path.join(
            data_path, "weights/", str(from_survey), "not_normalised/"
        )

        print(path_output)
        latest = tf.train.latest_checkpoint(path_output)
        net.load_weights(latest)

    # Define callbacks for VAE
    callbacks = define_callbacks("vae", survey_name)

    # Do the training for the VAE
    hist_vae = train_network(
        net,
        epochs,
        training_data_vae,
        validation_data_vae,
        batch_size,
        callbacks,
        verbose,
    )
    print("\nTraining of VAE done.")

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compilation of the deblender
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        metrics=["mse", kl_metric],
        experimental_run_tf_function=False,
    )
    print("\n\nDeblender model")
    net.summary()

    # Define callbacks for deblender
    callbacks = define_callbacks("deblender", survey_name)

    # Do the training for the deblender
    hist_deblender = train_network(
        net,
        epochs,
        training_data_deblender,
        validation_data_deblender,
        batch_size,
        callbacks,
        verbose,
    )
    print("\nTraining of Deblender done.")

    return hist_vae, hist_deblender, net
