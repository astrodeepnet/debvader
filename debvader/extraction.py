import numpy as np


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

            cutout_images[i] = field_image[x_start:x_end, y_start:y_end]
            list_idx.append(i)

        except ValueError:
            flag = True

    if flag:
        print(
            "Some galaxies are too close from the border of the field to be considered here."
        )

    return cutout_images, list_idx
