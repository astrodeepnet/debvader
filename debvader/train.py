import os

import pkg_resources
import tensorflow as tf
import tensorflow.keras.backend as K

from debvader import model
from debvader.metrics import vae_loss


def define_callbacks(weights_save_path, lr_scheduler_epochs=None):
    """
    Define callbacks for a network to train

    parameters:
        weights_save_path: path at which weights are to be saved.path at which weights are to be saved. By default, it saves weights in the data folder.
        lr_scheduler_epochs: number of iterations after which the learning rate is decreased by a factor of $e$.
            Default is None, and a constant learning rate is used
    """

    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_mse", "weights_noisy_v4.ckpt"),
        monitor="val_mse",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_loss", "weights_noisy_v4.ckpt"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    callbacks = [checkpointer_val_mse, checkpointer_val_loss]

    if lr_scheduler_epochs is not None:

        def scheduler(epoch, lr):
            if (epoch + 1) % lr_scheduler_epochs != 0:
                return lr
            else:
                return lr * tf.math.exp(-1.0)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks += [lr_scheduler]

    return callbacks


def train_network(
    net,
    epochs,
    training_data,
    validation_data,
    batch_size,
    train_encoder,
    train_decoder,
    weights_save_path=None,
    verbose=1,
    lr_scheduler_epochs=None,
):
    """
    train a network on data for a fixed number of epochs

    parameters:
        net: network to train
        epochs: number of epochs
        training_data: training data under the format of numpy arrays (inputs, labels)
        validation_data: validation data under the format of numpy arrays (inputs, labels)
        batch_size: size of batch for training
        train_encoder: boolean to select if encoder is to be trained.
        train_decoder: boolean to select if decoder is to be trained.
        weights_save_path: path at which weights are to be saved. By default, it saves weights in the data/trial folder.
        verbose: display of training (1:yes, 2: no)
        lr_scheduler_epochs: number of iterations after which the learning rate is decreased by a factor of $e$.
            Default is None, and a constant learning rate is used
    """
    if not ((lr_scheduler_epochs is None) | isinstance(lr_scheduler_epochs, int)):
        raise ValueError("lr_scheduler_epochs should either be 'None' or an int")

    if not (train_encoder or train_decoder):
        raise ValueError("At least one of encoder or decoder must be trainable.")

    net.get_layer("encoder").trainable = train_encoder
    net.get_layer("decoder").trainable = train_decoder

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

    print(net.summary())

    if weights_save_path is None:
        weights_save_path = pkg_resources.resource_filename("debvader", "data")
        weights_save_path = os.path.join(weights_save_path, "trial")

    callbacks = define_callbacks(
        weights_save_path, lr_scheduler_epochs=lr_scheduler_epochs
    )

    print("\nStart the training")
    if isinstance(training_data, tf.keras.utils.Sequence):
        hist = net.fit(
            x=training_data,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_data,
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=2,
        )

    else:
        hist = net.fit(
            x=training_data[0],
            y=training_data[1],
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            validation_data=(validation_data[0], validation_data[1]),
            validation_steps=int(len(validation_data[0]) / batch_size),
            callbacks=callbacks,
        )

    return hist


def train_deblender(
    training_data_vae,
    validation_data_vae,
    training_data_deblender,
    validation_data_deblender,
    epochs,
    input_shape=(59, 59, 6),
    latent_dim=32,
    filters=[32, 64, 128, 256],
    kernels=[3, 3, 3, 3],
    channel_last=True,
    batch_size=5,
    verbose=1,
    initial_weights_path=None,
    from_survey=None,
    weights_save_path=None,
    survey_name=None,
    lr_scheduler_epochs=None,
):
    """
    function to train a network for a new survey

    parameters:
        training_data_{}: training data under the format of numpy arrays (inputs, labels) for the vae or the deblender
        validation_data_{}: validation data under the format of numpy arrays (inputs, labels) for the vae or the deblender
        epochs: number of epochs of training
        input_shape: shape of input tensor, default value: (59, 59, 6)
        latent_dim: size of the latent space, default value:  32
        filters: filters used for the convolutional layers, default value: [32, 64, 128, 256]
        kernels: kernels used for the convolutional layers, default value: [3, 3, 3, 3]
        channel_last: boolean to indicate if the last lasat index for data refers to channels
        batch_size: size of batch for training
        verbose: display of training (1:yes, 2: no)
        initial_weights_path: path to folder from whrere initial weights are loaded for transfer learning.
            If the initial_weights_path is provided, the `from_survey` parameter will be ignored.
        from_survey: loads weights for transfer learning from corresponding folder within the `data` folder.
            Used only if `initial_weights_path` is None.
        weights_save_path: path at which weights are to be saved.
            If the path to save weights is passed, the `survey_name` parameter is ignored.
            By default, if both `weights_save_path` and `survey_name` is None, weights in the `trial` folder within the `data` folder
        survey_name: used to speficy the path within the `data/` folder where the trained weights are to be saved.
            By default, if both `weights_save_path` and `survey_name` is None, weights in the `trial` folder within the `data` folder
        lr_scheduler_epochs: number of iterations after which the learning rate is decreased by a factor of $e$.
            Default is None, and a constant learning rate is used
    """

    # Generate a network for training. The architecture is fixed.
    nb_of_bands = input_shape[-1]

    net, _, _, _ = model.create_model_vae(
        input_shape,
        latent_dim,
        filters,
        kernels,
    )

    # Check if data format is correct
    if not isinstance(training_data_vae, tf.keras.utils.Sequence):
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
    if (initial_weights_path is None) and (from_survey is not None):

        initial_weights_path = pkg_resources.resource_filename("debvader", "data")
        initial_weights_path = os.path.join(
            initial_weights_path, "weights", str(from_survey)
        )

    if initial_weights_path is not None:
        print("Initial training weights loaded from: " + initial_weights_path)
        latest = tf.train.latest_checkpoint(initial_weights_path)
        net.load_weights(latest)

    # train the VAE to learn representations of isolated galaxies.
    train_encoder = True
    train_decoder = True

    if weights_save_path is None:
        weights_save_path = pkg_resources.resource_filename("debvader", "data")

        sub_folder = survey_name
        if survey_name is None:
            sub_folder = "trial"

        weights_save_path = os.path.join(weights_save_path, "weights", sub_folder)

    vae_weights_path = os.path.join(weights_save_path, "vae")

    # Do the training for the VAE
    hist_vae = train_network(
        net=net,
        epochs=epochs,
        training_data=training_data_vae,
        validation_data=validation_data_vae,
        batch_size=batch_size,
        train_encoder=True,
        train_decoder=True,
        weights_save_path=vae_weights_path,
        verbose=verbose,
        lr_scheduler_epochs=lr_scheduler_epochs,
    )
    print("\nTraining of VAE done.")

    # Now, train the encoder as a deblender.

    # Set the decoder as non-trainable
    train_encoder = True
    train_decoder = False

    new_encoder = model.create_encoder(
        input_shape=input_shape, latent_dim=latent_dim, filters=filters, kernels=kernels
    )
    net.get_layer("encoder").set_weights(new_encoder.get_weights())

    deblender_weights_path = os.path.join(weights_save_path, "deblender")
    # Do the training for the deblender
    hist_deblender = train_network(
        net=net,
        epochs=epochs,
        training_data=training_data_deblender,
        validation_data=validation_data_deblender,
        batch_size=batch_size,
        train_encoder=train_encoder,
        train_decoder=train_decoder,
        weights_save_path=deblender_weights_path,
        verbose=verbose,
        lr_scheduler_epochs=lr_scheduler_epochs,
    )
    print("\nTraining of Deblender done.")

    return hist_vae, hist_deblender, net
