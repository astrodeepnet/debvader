# Import libraries
import numpy as np
import pkg_resources

from debvader.detection import detect_objects


def test_detection():
    # Load the field image
    data_folder_path = pkg_resources.resource_filename('debvader', "data/")
    field_img = np.load(data_folder_path + 'dc2_imgs/field/field_img.npy', mmap_mode = 'c')

    # Do detection: this funciton is using SExtractor
    galaxy_distances_to_center = detect_objects(field_img)

    # Check is object is not None
    assert galaxy_distances_to_center is not None
    
    # Check if the detection algorithm managed to find galaxies
    assert galaxy_distances_to_center.size!=0
