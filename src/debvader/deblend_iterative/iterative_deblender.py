import numpy as np

from debvader.deblend.field_deblender import DeblendField
from debvader.detect.detection import detect_objects
from debvader.training.metrics import mse


class IterativeDeblendField(DeblendField):
    def __init__(
        self,
        net,
        field_image,
        cutout_size=59,
        nb_of_bands=6,
        epistemic_uncertainty_estimation=False,
        normalise=False
    ): 
        super().__init__(net,field_image,cutout_size,nb_of_bands,epistemic_uncertainty_estimation,normalise)


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
            normalise: boolean to indicate if images need to be normalise
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
            # field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, galaxy_distances_to_center, mse_step = deblending_step(net, field_img_init, detection_up_to_k, cutout_images = None, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalise=normalise)
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

        # field_img_save, field_image, denoised_field, denoised_field_std, denoised_field_epistemic, cutout_images, output_images_mean, output_images_distribution, shifts, list_idx, nb_of_galaxies_in_deblended_field = deblend_field(net, field_image, detection_k, cutout_images = cutout_images, cutout_size = cutout_size, nb_of_bands = nb_of_bands, optimise_positions=optimise_positions, epistemic_uncertainty_estimation=epistemic_uncertainty_estimation, epistemic_criterion=epistemic_criterion, mse_criterion=mse_criterion, normalise=normalise)
        if len(res_step["list_idx"]) == 0:
            print("No more galaxies found")
            return self.res_deblend

        res_step["list_idx"] += (
            sum(self.nb_of_deblended_galaxies) - self.nb_of_deblended_galaxies[-1]
        )

        print(f"Deblend {self.nb_of_deblended_galaxies[-1]} more galaxy(ies)")
        # detection_confirmed = np.zeros((len(res_step["list_idx"]), 2))

        return res_step
