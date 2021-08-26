import os
import cv2


def detect_contours(filepath: str, threshold: int):
    # read image from filepath
    image = cv2.imread(os.path.normpath(filepath))

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=lambda x: len(x))
    largest_crv = contours[-1]

    return image, largest_crv


if __name__ == "__main__":
    thisfolder = os.path.dirname(os.path.realpath(__file__))
    fp = thisfolder + r"\demo_image.jpg"
    image, crv = detect_contours(fp, 170)

    # draw only largest contour
    image = cv2.drawContours(image, crv, -1, (0, 255, 0), 2)

    # show the images
    cv2.imshow("image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
