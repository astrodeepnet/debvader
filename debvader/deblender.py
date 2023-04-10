import numpy as np
import pandas as pd
import scipy
import tensorflow as tf

from debvader.detection import detect_objects
from debvader.extraction import extract_cutouts
from debvader.metrics import mse
from debvader.normalize import IdentityNorm, Normalizer
from debvader.optimization import position_optimization


def deblend(net, images, normalizer=None):
    """
    Deblend the image using the network
    parameters:
        net: neural network used to do the deblending
        images: array of images. It can contain only one image
        normalizer: object of debvader.normalize.Normalize, used to perform norm and denorm operations.
            Default value is set to debvader.normalize.IdentityNorm which does not perform any normalization
    """
    if normalizer is None:
        normalizer = IdentityNorm()

    if not isinstance(normalizer, Normalizer):
        raise ValueError(
            "The parameter `normalizer` should be an instance of debvader.normalize.Normalizer"
        )

    images = normalizer.forward(images)

    return net(tf.cast(images, tf.float32)).mean().numpy(), net(
        tf.cast(images, tf.float32)
    )


class DeblendField:
    def __init__(
        self,
        net,
        field_image,
        cutout_size=59,
        nb_of_bands=6,
        epistemic_uncertainty_estimation=False,
        normalizer=None,
    ):
        """
        to initialize

        parameters:
            net: network used to deblend the field
            field_image: image of the field to deblend
            cutout_size: size of the stamps
            nb_of_bands: number of filters in the image
            epistemic_uncertainty_estimation: boolean to indication if expestemic uncertainity extimation is to be done.
            normalizer: instance of debvader.normalize.Normalizer, used to perform normalization operations on input images.
                Default value is set to debvader.normalize.IdentityNorm which does not perform any normalization
        """

        self.net = net
        self.field_image = field_image.copy()  # TODO: proper garbage collection
        self.field_size = field_image.shape[1]
        self.cutout_size = cutout_size
        self.nb_of_bands = nb_of_bands
        self.epistemic_uncertainty_estimation = epistemic_uncertainty_estimation
        if normalizer is None:
            normalizer = IdentityNorm()

        if not isinstance(normalizer, Normalizer):
            raise ValueError(
                "The parameter `normalizer` should be an instance of debvader.normalize.Normalizer"
            )

        self.normalizer = normalizer
        self.nb_of_detected_objects = []
        self.nb_of_deblended_galaxies = []
        self.res_deblend = None
        self.mse = []
        self.optimize_positions = False

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

                # TODO: may be use some inbuild padding function?
                # subtract the image of the currently galaxy after positioning it correctly
                # TODO: try to avoid using scipy.ndimage.shift here, it is super slow. Diectly subtract at correct index instead of creating a padding?

                pos_offset = int((self.field_size - self.cutout_size) / 2)
                predicted_galaxy = isolated_galaxy_row["output_images_mean"].copy()

                if self.optimize_positions:

                    x_shift = isolated_galaxy_row["shifts"][0]
                    y_shift = isolated_galaxy_row["shifts"][1]

                    for band in range(self.nb_of_bands):
                        predicted_galaxy[:, :, band] = scipy.ndimage.shift(
                            predicted_galaxy[:, :, band],
                            shift=(x_shift, y_shift),
                        )

                x_start = int(
                    round(
                        isolated_galaxy_row["galaxy_distances_to_center_x"] + pos_offset
                    )
                )
                y_start = int(
                    round(
                        isolated_galaxy_row["galaxy_distances_to_center_y"] + pos_offset
                    )
                )
                deblended_image[
                    x_start : self.cutout_size + x_start,
                    y_start : self.cutout_size + y_start,
                ] -= isolated_galaxy_row["output_images_mean"]

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

                output_image_mean = isolated_galaxy_row["output_images_mean"]
                output_image_stddev = isolated_galaxy_row["output_images_stddev"]
                epistemic_uncertainty = isolated_galaxy_row["epistemic_uncertainty"]

                # First create padded images of the stamps at the size of the field to allow for a simple subtraction.

                if self.optimize_positions:

                    x_shift = isolated_galaxy_row["shifts"][0]
                    y_shift = isolated_galaxy_row["shifts"][1]

                    for band in range(
                        self.nb_of_bands
                    ):  # scipy.ndimage.shift is slow so looping over bands is faster

                        output_image_mean[:, :, band] += scipy.ndimage.shift(
                            output_image_mean[:, :, band], shift=(x_shift, y_shift)
                        )
                        output_image_stddev[:, :, band] += scipy.ndimage.shift(
                            output_image_stddev[:, :, band],
                            shift=(x_shift, y_shift),
                        )

                        if self.epistemic_uncertainty_estimation:
                            # Create the corresponding epistemic uncertainty image (aleatoric uncertainty).
                            epistemic_uncertainty = np.array(epistemic_uncertainty)
                            epistemic_uncertainty[:, :, band] += scipy.ndimage.shift(
                                epistemic_uncertainty[:, :, band],
                                shift=(x_shift, y_shift),
                            )

                pos_offset = int((self.field_size - self.cutout_size) / 2)

                x_start = int(
                    round(
                        isolated_galaxy_row["galaxy_distances_to_center_x"] + pos_offset
                    )
                )
                y_start = int(
                    round(
                        isolated_galaxy_row["galaxy_distances_to_center_y"] + pos_offset
                    )
                )

                denoised_field[
                    x_start : x_start + self.cutout_size,
                    y_start : y_start + self.cutout_size,
                ] += output_image_mean
                denoised_field_std[
                    x_start : x_start + self.cutout_size,
                    y_start : y_start + self.cutout_size,
                ] += output_image_stddev
                denoised_field_epistemic[
                    x_start : x_start + self.cutout_size,
                    y_start : y_start + self.cutout_size,
                ] += epistemic_uncertainty

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

    def calc_epistemic(self, cutout_image, output_image_mean):
        """
        Compute epistemic uncertainty (from the decoder of the deblender)

        parameters:
            cutout_image: image for which epistemic uncertainty is to be determined
            output_image_mean: mean of the output distribution predicted for each pixel
        """
        if self.epistemic_uncertainty_estimation:
            epistemic_uncertainty = np.std(
                deblend(
                    self.net,
                    np.array([cutout_image] * 100),
                    normalizer=self.normalizer,
                )[0],
                axis=0,
            )

            epistemic_uncertainty_normalised = np.sum(
                epistemic_uncertainty[:, :, 2]
            ) / np.sum(output_image_mean[:, :, 2])
        else:

            epistemic_uncertainty = list(
                np.zeros((self.cutout_size, self.cutout_size, self.nb_of_bands))
            )
            epistemic_uncertainty_normalised = 0

        return epistemic_uncertainty, epistemic_uncertainty_normalised

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
        self.optimize_positions = False

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

        # Deblend the cutouts around the detected galaxies. If needed, create the cutouts.
        if isinstance(cutout_images, np.ndarray):
            list_idx = list(range(0, len(cutout_images)))
        else:
            cutout_images, list_idx = extract_cutouts(
                field_image,
                self.field_size,
                galaxy_distances_to_center,
                self.cutout_size,
                self.nb_of_bands,
            )

        if list_idx == []:
            print("No galaxy deblended. End of the iterative procedure.")
            return res_deblend

        output_images_mean, output_images_distribution = deblend(
            self.net, cutout_images[list_idx], normalizer=self.normalizer
        )

        # Subtract each deblended galaxy to the field and add it to the denoised field.
        shifts = []
        galaxy_distances_to_center_x = []
        galaxy_distances_to_center_y = []
        epistemic_uncertainty = []
        passed_cuts = []

        for i, k in enumerate(list_idx):

            res_epistemic = self.calc_epistemic(cutout_images[k], output_images_mean[i])
            epistemic_uncertainty.append(res_epistemic[0])
            epistemic_uncertainty_normalised = res_epistemic[1]

            galaxy_distances_to_center_x.append(galaxy_distances_to_center[k][0])
            galaxy_distances_to_center_y.append(galaxy_distances_to_center[k][1])

            center_img_start = int(self.cutout_size / 2) - 5
            center_img_end = int(self.cutout_size / 2) + 5
            mse_center_img = mse(
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
                    field_image,
                    output_images_mean_padded,
                    galaxy_distances_to_center[k],
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

        res_deblend["output_images_mean"] = list(
            self.normalizer.backward(output_images_mean)
        )
        res_deblend["output_images_stddev"] = list(
            self.normalizer.backward(output_images_distribution.stddev().numpy())
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
        """

        # do the first step of deblending
        field_image = self.field_image.copy()
        res_step = self.deblending_step(
            field_image,
            galaxy_distances_to_center=galaxy_distances_to_center,
            cutout_images=cutout_images,
            optimise_positions=optimise_positions,
            epistemic_criterion=epistemic_criterion,
            mse_criterion=mse_criterion,
        )
        res_deblend = res_step

        new_residual_field = self.get_residual_field()
        self.mse += [mse(self.field_image, new_residual_field)]
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
            self.mse += [mse(prev_residual_field, new_residual_field)]

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
        galaxy_distances_to_center=None,
        cutout_images=None,
        optimise_positions=False,
        epistemic_criterion=100.0,
        mse_criterion=100.0,
    ):
        """
        One step of the iterative procedure called within iterative_procedure.

        parameters:
            field_image: image of the field to deblend
            galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
            cutout_images: stamps centered on the galaxies to deblend
            optimise_position: boolean to indicate if the user wants to use the scipy optimize package to optimise the position of the galaxy
            epistemic_criterion: cut for epistemic uncertainity to get rid of bad predictions
            mse_criterion: cut for mse_criterion to get rid of bad predictions
        """
        if galaxy_distances_to_center is None:
            galaxy_distances_to_center = detect_objects(field_image)
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
            galaxy_distances_to_center=galaxy_distances_to_center,
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
