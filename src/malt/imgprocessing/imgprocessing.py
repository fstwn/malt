# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np
import cv2


# FUNCTION DEFINITIONS --------------------------------------------------------

def capture_image():
    # set video device to external USB camera
    cap = cv2.VideoCapture(1)

    # settings for Logitech C930e
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")

    # if all is fine, read one image from the camera and return it
    ret, frame = cap.read()

    # while True:
    #     ret, frame = cap.read()
    #     cv2.imshow('Input', frame)

    #     c = cv2.waitKey(1)
    #     if c == 27:
    #         break

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

    return frame


def detect_contours_from_file(filepath: str,
                              thresh_binary: int,
                              thresh_area: float,
                              invert: bool = False):
    # read image from filepath
    image = cv2.imread(os.path.normpath(filepath))

    return detect_contours_from_image(image,
                                      thresh_binary,
                                      thresh_area,
                                      invert)


def detect_contours_from_image(image: np.ndarray,
                               thresh_binary: int,
                               thresh_area: float,
                               invert: bool = False):
    if invert:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    # flip image to avoid mirrored contours
    image = cv2.flip(image, 0)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, thresh_binary, 255, threshold_type)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area and filter against threshold
    if thresh_area > 0:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = [contours[i] for i in range(len(contours))
                    if areas[i] > thresh_area]

    return image, contours


# TEST MODULE -----------------------------------------------------------------

def test_detect_contours_from_file():
    # use the demo image to perform some contour detection
    thisfolder = os.path.dirname(os.path.realpath(__file__))
    fp = thisfolder + r"\demo_image.jpg"
    image, crvs = detect_contours_from_file(fp, 170, 50.0)

    # draw only largest contour
    image = cv2.drawContours(image, crvs, -1, (0, 255, 0), 2)

    # show the image
    cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_detect_contours_from_image():
    # capture an image to perform some contour detection
    image, crvs = detect_contours_from_image(capture_image(), 170, 50.0, False)

    # draw only largest contour
    image = cv2.drawContours(image, crvs, -1, (0, 255, 0), 2)

    # show the image
    cv2.imshow("Detected Contours in Captured Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test contour detection
    # test_detect_contours()
    # capture_image()
    test_detect_contours_from_image()
