# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np
import cv2


# FUNCTION DEFINITIONS --------------------------------------------------------

def capture_image(device: int = 0):
    # set video device to external USB camera
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)

    # settings for Logitech C930e
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam!")

    # if all is fine, read one image from the camera and return it
    ret, frame = cap.read()

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

    return frame


def calibrate_camera(device: int = 0,
                     dwidth: int = 3780,
                     dheight: int = 1890,
                     showresult: bool = False):

    # capture an image with the camera
    image = capture_image(device)
    image_copy = image.copy()

    # define window name
    wname = "Pick four Points for Calibration, then press Enter."

    # define click event function
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image_copy, (x, y), 4, (0, 0, 255), -1)
            points.append([x, y])
            if len(points) <= 4:
                cv2.imshow(wname, image_copy)

    # define point sorting clockwise for picked points
    def sort_pts(points):
        sorted_pts = np.zeros((4, 2), dtype="float32")
        s = np.sum(points, axis=1)
        sorted_pts[0] = points[np.argmin(s)]
        sorted_pts[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        sorted_pts[1] = points[np.argmin(diff)]
        sorted_pts[3] = points[np.argmax(diff)]

        return sorted_pts

    # display image and let user pick points
    points = []

    cv2.imshow(wname, image_copy)
    cv2.setWindowProperty(wname, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(wname, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # sort the points clockwise
    sorted_pts = sort_pts(points)

    # define destination image based on table size
    dst_image = 255 * np.zeros(shape=[int(dheight), int(dwidth), 3],
                               dtype=np.uint8)

    # extract source and destination points
    h_dst, w_dst, c_dst = dst_image.shape
    src_pts = np.float32(sorted_pts)
    dst_pts = np.float32([[0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]])

    # compute transformation matrix
    xform = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # show the warped image to the user
    if showresult:
        warped_img = cv2.warpPerspective(image, xform, (w_dst, h_dst))
        cv2.imshow("Warped Image", warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return xform


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
    # CHAIN_APPROX_TC89_L1
    # CHAIN_APPROX_TC89_KCOS
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by area and filter against threshold
    if thresh_area > 0:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = [contours[i] for i in range(len(contours))
                    if areas[i] > thresh_area]

    return image, contours


def warp_image(image, xform, width, height):
    """Wrapper for cv2.warpPerspective"""
    return cv2.warpPerspective(image, xform, (width, height))


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
    # test_detect_contours_from_image()

    calibrate_camera(1)
