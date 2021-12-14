from debvader.debvader import extract_cutouts
import numpy as np

def test_cutouts_border():
    
    field_size=15
    nb_of_bands = 3

    image = np.random.rand(1, field_size, field_size, nb_of_bands)
    galaxy_distances_to_center = [[5,5]]
    cutout_size = 5

    cutout = extract_cutouts(field_image=image.copy(), field_size=15, galaxy_distances_to_center=galaxy_distances_to_center, cutout_size=cutout_size, nb_of_bands=nb_of_bands)

    assert cutout[0][0][1][2][0] == image[0][11][12][0]
    assert cutout[0][0][4][4][2] == image[0][14][14][2]
    assert cutout[0][0][2][3][1] == image[0][12][13][1]

    galaxy_distances_to_center = [[-5,-5]]
    cutout_size = 5

    cutout = extract_cutouts(field_image=image.copy(), field_size=15, galaxy_distances_to_center=galaxy_distances_to_center, cutout_size=cutout_size, nb_of_bands=nb_of_bands)

    assert cutout[0][0][4][4][0] == image[0][4][4][0]
    assert cutout[0][0][2][3][1] == image[0][2][3][1]
    assert cutout[0][0][1][2][2] == image[0][1][2][2]

    # test when cutout is too large
    
    galaxy_distances_to_center = [[6,6]]
    cutout_size = 5

    cutout = extract_cutouts(field_image=image.copy(), field_size=15, galaxy_distances_to_center=galaxy_distances_to_center, cutout_size=cutout_size, nb_of_bands=nb_of_bands)

    assert len(cutout[1]) == 0
