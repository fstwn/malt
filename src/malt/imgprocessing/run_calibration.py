# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import imgprocessing


# MAIN ROUTINE ----------------------------------------------------------------

if __name__ == "__main__":
    # run the calibration routine on the chessboard images
    # chessboard images have to be placed in the "imgs_chessboard" directory!
    imgprocessing.compute_camera_coefficients()
