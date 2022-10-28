# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import glob
import os

# THIRD PARTY MODULE IMPORTS --------------------------------------------------

import cv2


# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import imgprocessing
from malt.hopsutilities import sanitize_path


# MAIN ROUTINE ----------------------------------------------------------------

if __name__ == "__main__":

    # get location of this file
    HERE = os.path.dirname(sanitize_path(__file__))
    # directory of raw images before undistortion
    raw_dir = sanitize_path(os.path.join(HERE, "imgs_raw"))
    # directory to save resulting, undistorted images
    undistorted_dir = sanitize_path(os.path.join(HERE, "imgs_undistorted"))

    # load coefficients from previously saved file
    print("[OPENCV] Loading camra coefficients from file...")
    mtx, dist = imgprocessing.load_coefficients(sanitize_path(os.path.join(
                                                    HERE, "coefficients.yml")))

    # apply the camera matrix to all target images
    print("[OPENCV] Applying undistortion to all raw images...")

    # get all scan images to apply undistortion to...
    scan_imgs = glob.glob(os.path.join(raw_dir, "*.jpg"))

    for i, img in enumerate(scan_imgs):
        print("[OPENCV] Undistorting image {0} of {1}".format(i + 1,
                                                              len(scan_imgs)))
        # apply undistortion
        undistorted_img = imgprocessing.undistort_image(cv2.imread(img),
                                                        mtx,
                                                        dist)
        # write undistorted image to new file
        cv2.imwrite(sanitize_path(os.path.join(undistorted_dir,
                                  os.path.split(img)[1])), undistorted_img)
