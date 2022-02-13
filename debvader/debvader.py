import os

import numpy as np
import pandas as pd
import pkg_resources
import scipy
import sep
import tensorflow as tf
from scipy import optimize
from skimage import metrics

from debvader import model


def extract_cutouts(
    field_image, field_size, galaxy_distances_to_center, cutout_size=59, nb_of_bands=6
):
    """
    Extract the cutouts around particular galaxies in the field
    parameters:
        field_image: image of the field to deblend
        field_size: size of the field
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_size: size of the stamps
    """
    cutout_images = np.zeros(
        (len(galaxy_distances_to_center), cutout_size, cutout_size, nb_of_bands)
    )
    list_idx = []
    flag = False

    for i in range(len(galaxy_distances_to_center)):
        try:
            x_shift = galaxy_distances_to_center[i][0]
            y_shift = galaxy_distances_to_center[i][1]

            x_start = -int(cutout_size / 2) + int(x_shift) + int(field_size / 2)
            x_end = int(cutout_size / 2) + int(x_shift) + int(field_size / 2) + 1

            y_start = -int(cutout_size / 2) + int(y_shift) + int(field_size / 2)
            y_end = int(cutout_size / 2) + int(y_shift) + int(field_size / 2) + 1

            cutout_images[i] = field_image[0, x_start:x_end, y_start:y_end]
            list_idx.append(i)

        except ValueError:
            flag = True

    if flag:
        print(
            "Some galaxies are too close from the border of the field to be considered here."
        )

    return cutout_images, list_idx


def detect_objects(field_image):
    """
    Detect the objects in the field_image image using the SExtractor detection algorithm.
    test for dev branch
    """
    field_image = field_image.copy()
    field_size = field_image.shape[1]
    galaxy_distances_to_center = []

    r_band_data = field_image[0, :, :, 2].copy()
    bkg = sep.Background(r_band_data)

    r_band_foreground = r_band_data - bkg

    DETECT_THRESH = 0.8
    deblend_cont = 0.00001
    deblend_nthresh = 64
    minarea = 4
    filter_type = "conv"
    # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
    filter_kernel = np.array(
        [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
        ]
    )

    objects = sep.extract(
        data=r_band_foreground,
        thresh=DETECT_THRESH,
        err=bkg.globalrms,
        deblend_cont=deblend_cont,
        deblend_nthresh=deblend_nthresh,
        minarea=minarea,
        filter_kernel=filter_kernel,
        filter_type=filter_type,
    )

    for i in range(len(objects["y"])):
        galaxy_distances_to_center.append(
            (
                np.round(-int(field_size / 2) + objects["y"][i]),
                np.round(-int(field_size / 2) + objects["x"][i]),
            )
        )

    return np.array(galaxy_distances_to_center)


def load_deblender(
    survey, input_shape, latent_dim, filters, kernels, return_encoder_decoder_z=False
):
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
        # xent_loss = K.mean(
        #    K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1, 2, 3])
        # )
        return -x_decoded_mean.log_prob(x)  # xent_loss

    # Set the decoder as non-trainable
    decoder.trainable = False

    # Compile the model
    net.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=vae_loss,
        experimental_run_tf_function=False,
    )

    # Load the weights corresponding to the chosen survey
    data_path = pkg_resources.resource_filename("debvader", "data/")
    loading_path = os.path.join(data_path, "weights/", survey, "not_normalised/loss/")
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)

    if return_encoder_decoder_z:
        return net, encoder, decoder, z
    else:
        return net


def deblend(net, images, normalised=False):
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

    return net(tf.cast(images, tf.float32)).mean().numpy(), net(
        tf.cast(images, tf.float32)
    )


def position_optimization(
    field_image,
    output_image_mean_padded,
    galaxy_distance_to_center,
    method="scipy-minimize",
):
    """
    Find shifts in the position of the deblended galaxy to minimize the mse between field_image

    parameters:
        field image: image of the entire field of galaxy to be deblended.
        output_images_mean_padded: predicted image of the galaxy that is to be optimized.
        galaxy_distances_to_center: distance of the predicted galaxy from the center, as detected by the detection algorithm.
    """

    assert method in ["scipy-minimize"]

    if method == "scipy-minimize":

        def fun(x, img, net_output):
            """
            parameters:
                x: shifts for x and y position
                img: field image
                net_output: predicted image if the galaxy
            """
            return metrics.mean_squared_error(
                img, scipy.ndimage.shift(net_output, shift=(x[0], x[1]))
            )

        r_band_field = field_image[:, :, 2]
        r_band_perdiction = output_image_mean_padded[:, :, 2]
        opt = optimize.least_squares(
            fun,
            (0.0, 0.0),
            args=(
                r_band_field,
                scipy.ndimage.shift(
                    r_band_perdiction,
                    shift=(galaxy_distance_to_center[0], galaxy_distance_to_center[1]),
                ),
            ),
            bounds=(-3, 3),
        )

        shift_x = opt.x[0]
        shift_y = opt.x[1]

    return shift_x, shift_y


class DeblendField:
    def __init__(
        self,
        net,
        field_image,
        cutout_size=59,
        nb_of_bands=6,
        epistemic_uncertainty_estimation=False,
        normalised=False,
    ):
        """
        to initialize

        parameters:
            net: network used to deblend the field
            field_image: image of the field to deblend
            cutout_size: size of the stamps
            nb_of_bands: number of filters in the image
            epistemic_uncertainty_estimation: boolean to indication if expestemic uncertainity extimation is to be done.
            normalised: boolean to indicate if images need to be normalised
        """

        self.net = net
        self.field_image = field_image.copy()  # TODO: proper garbage collection
        self.field_size = field_image.shape[1]
        self.cutout_size = cutout_size
        self.nb_of_bands = nb_of_bands
        self.epistemic_uncertainty_estimation = epistemic_uncertainty_estimation
        self.normalised = normalised
        self.nb_of_detected_objects = []
        self.nb_of_deblended_galaxies = []
        self.res_deblend = None
        self.mse = []

    def get_residual_field(self, res_deblend=None):
        """
        Calculates the residual field

        parameters:
            res_deblend: np.recarray that takes as input the result of deblending.
                if left as None, it will automatically use the output of deblend_field
                or iterative_deblending function
        returns:
            deblended_image: residual image after substracting the predicted deblended images.
                dimensions are the same as the input field.
        """

        if res_deblend is None:
            res_deblend = self.res_deblend

        deblended_image = self.field_image.copy()
        if res_deblend is not None:
            for isolated_galaxy_row in res_deblend:

                # First create padded images of the stamps at the size of the field to allow for a simple subtraction.

                # TODO: may be use some inbuild padding function?
                output_images_mean_padded = np.zeros(
                    (1, self.field_size, self.field_size, self.nb_of_bands)
                )
                pos_offset = int((self.field_size - self.cutout_size) / 2)

                output_images_mean_padded[
                    :,
                    pos_offset : self.cutout_size + pos_offset,
                    pos_offset : self.cutout_size + pos_offset,
                    :,
                ] = isolated_galaxy_row["output_images_mean"]

                # subtract the image of the currently galaxy after positioning it correctly
                # TODO: try to avoid using scipy.ndimage.shift here, it is super slow. Diectly subtract at correct index instead of creating a padding?
                x_pos = (
                    isolated_galaxy_row["galaxy_distances_to_center_x"]
                    + isolated_galaxy_row["shifts"][0]
                )
                y_pos = (
                    isolated_galaxy_row["galaxy_distances_to_center_y"]
                    + isolated_galaxy_row["shifts"][1]
                )

                for band in range(self.nb_of_bands):
                    deblended_image[0, :, :, band] -= scipy.ndimage.shift(
                        output_images_mean_padded[0, :, :, band], shift=(x_pos, y_pos)
                    )

        return deblended_image

    def get_predicted_field(self, res_deblend=None):
        """
        Calculates the predicted mean field, predicted stddev field and epistemic uncertainity field.

        parameters:
            res_deblend: np.recarray that takes as input the result of deblending.
                if left as None, it will automatically use the output of deblend_field or iterative_deblending functions

        returns:
            predicted_field: dictionary with keys - 'predicted_mean_field', 'predicted_stddev_field' and 'predicted_epistemic_field'
        """
        if res_deblend is None:
            res_deblend = self.res_deblend

        denoised_field = np.zeros((self.field_size, self.field_size, self.nb_of_bands))
        denoised_field_std = np.zeros(
            (self.field_size, self.field_size, self.nb_of_bands)
        )
        denoised_field_epistemic = np.zeros(
            (self.field_size, self.field_size, self.nb_of_bands)
        )

        if res_deblend is not None:

            for isolated_galaxy_row in res_deblend:

                output_images_mean = isolated_galaxy_row["output_images_mean"]
                output_images_stddev = isolated_galaxy_row["output_images_stddev"]
                epistemic_uncertainty = isolated_galaxy_row["epistemic_uncertainty"]

                # First create padded images of the stamps at the size of the field to allow for a simple subtraction.

                pad_start = int((self.field_size - self.cutout_size) / 2)
                pad_end = self.cutout_size + int(
                    (self.field_size - self.cutout_size) / 2
                )

                output_images_mean_padded = np.zeros(
                    (self.field_size, self.field_size, self.nb_of_bands)
                )
                output_images_mean_padded[
                    pad_start:pad_end, pad_start:pad_end, :
                ] = output_images_mean

                # Create the corresponding standard deviation image (aleatoric uncertainty).
                output_images_stddev_padded = np.zeros(
                    (self.field_size, self.field_size, self.nb_of_bands)
                )
                output_images_stddev_padded[
                    pad_start:pad_end, pad_start:pad_end, :
                ] = output_images_stddev

                x_pos = (
                    isolated_galaxy_row["galaxy_distances_to_center_x"]
                    + isolated_galaxy_row["shifts"][0]
                )
                y_pos = (
                    isolated_galaxy_row["galaxy_distances_to_center_y"]
                    + isolated_galaxy_row["shifts"][1]
                )

                for band in range(
                    self.nb_of_bands
                ):  # scipy.ndimage.shift is slow so looping over bands is faster

                    denoised_field[:, :, band] += scipy.ndimage.shift(
                        output_images_mean_padded[:, :, band], shift=(x_pos, y_pos)
                    )
                    denoised_field_std[:, :, band] += scipy.ndimage.shift(
                        output_images_stddev_padded[:, :, band], shift=(x_pos, y_pos)
                    )

                    output_images_epistemic_padded = np.zeros(
                        (self.field_size, self.field_size, self.nb_of_bands)
                    )
                    if self.epistemic_uncertainty_estimation:
                        # Create the corresponding epistemic uncertainty image (aleatoric uncertainty).
                        output_images_epistemic_padded[
                            pad_start:pad_end, pad_start:pad_end, :
                        ] = np.array(epistemic_uncertainty)
                        denoised_field_epistemic[:, :, band] += scipy.ndimage.shift(
                            output_images_epistemic_padded[:, :, band],
                            shift=(x_pos, y_pos),
                        )

        predicted_field = {}
        predicted_field["predicted_mean_field"] = denoised_field
        predicted_field["predicted_stddev_field"] = denoised_field_std
        predicted_field["predicted_epistemic_field"] = denoised_field_epistemic

        return predicted_field

    def get_deblending_meta_data(self, res_deblend=None):
        """
        function to compute: the residual image, predicted mean field, preicted stddev field and predicted_epistemic_field at the same time

        parameters:
            res_deblend: np.recarray that takes as input the result of deblending.
                if left as None, it will automatically use the output of deblend_field or iterative_deblending functions

        returns:
            predicted_field: dictionary with keys - 'field_image', 'delended_image', 'predicted_mean_field', 'predicted_stddev_field'
                and 'predicted_epistemic_field'
        """
        res_deblend_meta = {}
        res_deblend_meta["field_image"] = self.field_image
        res_deblend_meta["deblended_image"] = self.get_residual_field(res_deblend)
        predicted_field = self.get_predicted_field(res_deblend)
        res_deblend_meta["predicted_mean_field"] = predicted_field[
            "predicted_mean_field"
        ]
        res_deblend_meta["predicted_stddev_field"] = predicted_field[
            "predicted_stddev_field"
        ]
        res_deblend_meta["predicted_epistemic_field"] = predicted_field[
            "predicted_epistemic_field"
        ]

        return res_deblend_meta

    def deblend_field(
        self,
        galaxy_distances_to_center,
        cutout_images=None,
        optimise_positions=False,
        epistemic_criterion=100.0,
        mse_criterion=100.0,
        field_image=None,
    ):
        """
        Deblend a field of galaxies it returns
        parameters:
            galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
            cutout_images: stamps centered on the galaxies to deblend
            optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
            epistemic_criterion: cut for epistemic uncertainity to get rid of bad predictions
            mse_criterion: cut for mse_criterion to get rid of bad predictions
            field_image: image of the field to deblend. Recommended to leave as None, by default it will use self.field_image.
                Note that providing a field other than self.field_image may lead to incorrect results on the residual image.

        returns:
            self.res_deblended: np.recarray containing the following for each detected galaxy -
                cutout_images, output_images_mean, output_images_stddev, shifts, list_idx, galaxy_distances_to_center_x, galaxy_distances_to_center_y
        """

        res_deblend = dict()
        res_deblend["cutout_images"] = None
        res_deblend["output_images_mean"] = None
        res_deblend["output_images_stddev"] = None
        res_deblend["shifts"] = None
        res_deblend["list_idx"] = None

        if (
            field_image is None
        ):  # TODO: if the user passes a differnet field image instead of self.field it will mess up the self.get_residual (and others) function
            field_image = self.field_image.copy()

        field_size = field_image.shape[1]

        # Deblend the cutouts around the detected galaxies. If needed, create the cutouts.
        if isinstance(cutout_images, np.ndarray):
            output_images_mean, output_images_distribution = deblend(
                self.net, cutout_images, normalised=self.normalised
            )
            list_idx = list(range(0, len(output_images_mean)))
        else:
            cutout_images, list_idx = extract_cutouts(
                field_image,
                field_size,
                galaxy_distances_to_center,
                self.cutout_size,
                self.nb_of_bands,
            )
            output_images_mean, output_images_distribution = deblend(
                self.net, cutout_images[list_idx], normalised=self.normalised
            )
        if list_idx == []:
            print("No galaxy deblended. End of the iterative procedure.")
            return res_deblend

        # Subtract each deblended galaxy to the field and add it to the denoised field.
        shifts = []
        galaxy_distances_to_center_x = []
        galaxy_distances_to_center_y = []

        passed_cuts = []
        if self.epistemic_uncertainty_estimation:
            epistemic_uncertainty = []

        else:
            epistemic_uncertainty = list(
                np.zeros(
                    (
                        len(list_idx),
                        self.cutout_size,
                        self.cutout_size,
                        self.nb_of_bands,
                    )
                )
            )

        for i, k in enumerate(list_idx):

            # Compute epistemic uncertainty (from the decoder of the deblender)
            if self.epistemic_uncertainty_estimation:
                epistemic_uncertainty.append(
                    np.std(
                        deblend(
                            self.net,
                            np.array([cutout_images[k]] * 100),
                            normalised=self.normalised,
                        )[0],
                        axis=0,
                    )
                )
                epistemic_uncertainty_normalised = np.sum(
                    epistemic_uncertainty[i][:, :, 2]
                ) / np.sum(output_images_mean[i, :, :, 2])
            else:
                epistemic_uncertainty_normalised = 0

            galaxy_distances_to_center_x.append(galaxy_distances_to_center[k][0])
            galaxy_distances_to_center_y.append(galaxy_distances_to_center[k][1])

            center_img_start = int(self.cutout_size / 2) - 5
            center_img_end = int(self.cutout_size / 2) + 5
            mse_center_img = metrics.mean_squared_error(
                cutout_images[
                    k, center_img_start:center_img_end, center_img_start:center_img_end
                ],
                output_images_mean[
                    i, center_img_start:center_img_end, center_img_start:center_img_end
                ],
            )

            shift_x = 0
            shift_y = 0

            if optimise_positions:  # TODO: Check if this is giving funny results
                output_images_mean_padded = np.zeros(
                    (self.field_size, self.field_size, self.nb_of_bands)
                )
                output_images_mean_padded[
                    int((self.field_size - self.cutout_size) / 2) : self.cutout_size
                    + int((self.field_size - self.cutout_size) / 2),
                    int((self.field_size - self.cutout_size) / 2) : self.cutout_size
                    + int((self.field_size - self.cutout_size) / 2),
                    :,
                ] = output_images_mean[i]
                shift_x, shift_y = position_optimization(
                    field_image[0],
                    output_images_mean_padded,
                    galaxy_distances_to_center[k],
                    method="scipy-minimize",
                )

            shifts.append(np.array([shift_x, shift_y]))

            if (epistemic_uncertainty_normalised > epistemic_criterion) or (
                mse_center_img > mse_criterion
            ):  # avoid to add galaxies generated with too high uncertainty
                passed_cuts.append(False)
            else:
                passed_cuts.append(True)

        self.nb_of_detected_objects += [len(list(galaxy_distances_to_center))]
        self.nb_of_deblended_galaxies += [len(list_idx)]

        res_deblend["cutout_images"] = list(cutout_images[list_idx])
        res_deblend["output_images_mean"] = list(output_images_mean)
        res_deblend["output_images_stddev"] = list(
            output_images_distribution.stddev().numpy()
        )
        res_deblend["shifts"] = shifts
        res_deblend["list_idx"] = list_idx
        res_deblend["galaxy_distances_to_center_x"] = galaxy_distances_to_center_x
        res_deblend["galaxy_distances_to_center_y"] = galaxy_distances_to_center_y

        # TODO: Return the lines below only if epistemic uncertainity esimation is true (reduce memory overhead)
        res_deblend["epistemic_uncertainty"] = epistemic_uncertainty
        res_deblend["passed_cuts"] = passed_cuts

        self.res_deblend = pd.DataFrame(res_deblend).to_records(index=False)

        return self.res_deblend

    def iterative_deblending(
        self,
        galaxy_distances_to_center=None,
        cutout_images=None,
        optimise_positions=False,
        epistemic_criterion=100.0,
        mse_criterion=100.0,
    ):
        """
        Do the iterative deblending of a scene
        paramters:
            galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
            cutout_images: stamps centered on the galaxies to deblend
            optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
            epistemic_criterion: cut for epistemic uncertainity to get rid of bad predictions
            mse_criterion: cut for mse_criterion to get rid of bad predictions
            normalised: boolean to indicate if images need to be normalised
        """

        # do the first step of deblending
        field_image = self.field_image.copy()
        res_step = self.deblending_step(
            field_image,
            cutout_images=cutout_images,
            optimise_positions=optimise_positions,
            epistemic_criterion=epistemic_criterion,
            mse_criterion=mse_criterion,
        )
        res_deblend = res_step

        new_residual_field = self.get_residual_field()
        self.mse += [metrics.mean_squared_error(self.field_image, new_residual_field)]
        shifts_previous = []
        k = 1
        diff_mse = -1

        # Now iterate over
        while len(res_step["shifts"]) > len(shifts_previous):

            print(f"iteration {k}")
            shifts_previous = res_step["shifts"]

            prev_residual_field = new_residual_field

            # deblending step will run detection and deblending on the residual field
            res_step = self.deblending_step(
                prev_residual_field,
                cutout_images=None,
                optimise_positions=optimise_positions,
                mse_criterion=mse_criterion,
            )

            # compute the MSE after this iteration step
            new_residual_field = self.get_residual_field()
            self.mse += [
                metrics.mean_squared_error(prev_residual_field, new_residual_field)
            ]
            # field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step = deblending_step(net, field_img_init, detection_up_to_k, cutout_images = None, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)
            # field_img_init=field_img_save.copy()

            if res_step["list_idx"] is None:
                break

            res_deblend = np.concatenate([res_deblend, res_step])
            k += 1

            print(
                f"{sum(self.nb_of_deblended_galaxies)} galaxies found up to this step."
            )
            print(
                f"deta_mse = {diff_mse}, mse_iteration = "
                + str(self.mse[-1])
                + " and mse_previous_step = "
                + str(self.mse[-2])
            )

        print("converged !")

        self.res_deblend = res_deblend

        return self.res_deblend

    def deblending_step(
        self,
        field_image,
        cutout_images=None,
        optimise_positions=False,
        epistemic_criterion=100.0,
        mse_criterion=100.0,
    ):
        """
        One step of the iterative procedure called within iterative_procedure.

        paramters:
            field_image: image of the field to deblend
            cutout_images: stamps centered on the galaxies to deblend
            optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
            epistemic_criterion: cut for epistemic uncertainity to get rid of bad predictions
            mse_criterion: cut for mse_criterion to get rid of bad predictions
        """
        detection_k = detect_objects(field_image)
        # Avoid to have several detection at the same location

        # TODO: Fix this part to get rid of false detections.
        # Ideally call a remove_residual_detection function which can be developed later!

        # if isinstance(galaxy_distances_to_center_total, np.ndarray):
        #    for i in range (len(detection_k)):
        #        if detection_k[i] in galaxy_distances_to_center_total: TODO: Does this make sense??
        #            idx_to_remove.append(i)
        #    detection_k = np.delete(detection_k, idx_to_remove, axis = 0)

        res_step = self.deblend_field(
            field_image=field_image,
            galaxy_distances_to_center=detection_k,
            cutout_images=cutout_images,
            optimise_positions=optimise_positions,
            epistemic_criterion=epistemic_criterion,
            mse_criterion=mse_criterion,
        )

        # field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, list_idx, nb_of_galaxies_in_deblended_field = deblend_field(net, field_image, detection_k, cutout_images = cutout_images, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalised=normalised)
        if len(res_step["list_idx"]) == 0:
            print("No more galaxies found")
            return self.res_deblend

        res_step["list_idx"] += (
            sum(self.nb_of_deblended_galaxies) - self.nb_of_deblended_galaxies[-1]
        )

        print(f"Deblend {self.nb_of_deblended_galaxies[-1]} more galaxy(ies)")
        # detection_confirmed = np.zeros((len(res_step["list_idx"]), 2))

        return res_step
