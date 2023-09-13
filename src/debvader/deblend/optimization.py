import numpy as np
import scipy
from scipy import optimize


def position_optimization(
    field_image,
    output_image_mean_padded,
    galaxy_distance_to_center,
):
    """
    Find shifts in the position of the deblended galaxy to minimize the mse between field_image

    parameters:
        field image: image of the entire field of galaxy to be deblended.
        output_images_mean_padded: predicted image of the galaxy that is to be optimized.
        galaxy_distances_to_center: distance of the predicted galaxy from the center, as detected by the detection algorithm.
    """

    def fun(x, img, net_output):
        """
        parameters:
            x: shifts for x and y position
            img: field image
            net_output: predicted image if the galaxy
        """

        mse = np.square(
            img - scipy.ndimage.shift(net_output, shift=(x[0], x[1]))
        ).mean()

        return mse

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
