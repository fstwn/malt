# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import cv2


# FUNCTION DEFINITIONS --------------------------------------------------------

def detect_contours(filepath: str, thresh_binary: int, thresh_area: float,
                    invert: bool=False):
    if invert:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    # read image from filepath and flip it to avoid mirrored contours
    image = cv2.imread(os.path.normpath(filepath))
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

if __name__ == "__main__":
    # use the demo image to perform some contour detection
    thisfolder = os.path.dirname(os.path.realpath(__file__))
    fp = thisfolder + r"\demo_image.jpg"
    image, crvs = detect_contours(fp, 170, 50.0)

    # draw only largest contour
    image = cv2.drawContours(image, crvs, -1, (0, 255, 0), 2)

    # show the image
    cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
