import numpy as np
import sep


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

    DETECT_THRESH = 1.5
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
