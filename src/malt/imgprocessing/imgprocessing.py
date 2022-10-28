# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import glob
import os

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np
import cv2

# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt.hopsutilities import sanitize_path


# FUNCTION DEFINITIONS --------------------------------------------------------

def capture_image(device: int = 0):
    """Capture an image using a connected camera and return the frame."""
    # set video device to external USB camera
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)

    # settings for Logitech C930e
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("[OPENCV] Cannot open camera!")

    # if all is fine, read one image from the camera and return it
    ret, frame = cap.read()

    # release the camera
    cap.release()
    cv2.destroyAllWindows()

    return frame


def calibrate_camera_image(image,
                           dwidth: int = 1000,
                           dheight: int = 1000,
                           showresult: bool = False):

    # create image copy
    image_copy = image.copy()

    # compute scaled display size
    ih, iw = image_copy.shape[:2]
    displaysize = 1000
    if ih > iw and ih > displaysize:
        rsf = displaysize / ih
    elif iw >= ih and iw > displaysize:
        rsf = displaysize / iw
    else:
        rsf = 1.0

    # define window name
    wname = "Pick four Points for Calibration, then press Enter."

    # define click event function
    def click_event(event, x, y, flags, param):
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

    cv2.namedWindow(wname, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, int(iw * rsf), int(ih * rsf))
    cv2.setWindowProperty(wname, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(wname, click_event)

    while True:
        cv2.imshow(wname, image_copy)
        cv2.waitKey(0)
        break

    cv2.destroyWindow(wname)
    cv2.destroyAllWindows()

    # sort the points clockwise
    sorted_pts = sort_pts(points)

    # pixels per inch
    ppi = 96
    # millimeters per inch
    mmpi = 25.4

    # dheight and dwidth in millimeters
    dheight = (dheight / mmpi) * ppi
    dwidth = (dwidth / mmpi) * ppi

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


def calibrate_camera_file(filepath,
                          dwidth: int = 3780,
                          dheight: int = 1890,
                          showresult: bool = False):

    # read image from filepath
    image = read_image(filepath)

    return calibrate_camera_image(image, dwidth, dheight, showresult)


def calibrate_chessboard(images,
                         width: int = 7,
                         height: int = 9,
                         squaresize: float = 2.0,
                         displaysize: int = 1000):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for the chessboard, depending on width and height
    # like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2)

    # multiplicate points with square size in cm
    objp = objp * squaresize

    # Arrays to store object points and image points from all the images.
    # objpoints: 3d point in real world space
    objpoints = []
    # imgpoints: 2d points in image plane.
    imgpoints = []

    if not images:
        print("[OPENCV] No images found for chessboard calibration!")
        return

    # loop through test images and determine imgpoints
    print("[OPENCV] Determining object points for camera calibration...")
    for i, fname in enumerate(images):
        print("[OPENCV] Processing image {0} of {1}...".format(i + 1,
                                                               len(images)))
        # read image and convert to grayscale
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # compute display size for image
        ih, iw = img.shape[:2]
        if ih > iw and ih > displaysize:
            rsf = displaysize / ih
        elif iw >= ih and iw > displaysize:
            rsf = displaysize / iw
        else:
            rsf = 1.0

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (height, width), None)

        # if found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,
                                        corners,
                                        (11, 11),
                                        (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

            # draw and display the corners
            cv2.drawChessboardCorners(img, (height, width), corners2, ret)
            windowname = "Found object points in image"
            cv2.namedWindow(windowname, flags=cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowname, int(iw * rsf), int(ih * rsf))
            cv2.imshow(windowname, img)
            cv2.waitKey(500)
        else:
            print("[OPENCV] Chessboard corners could not be found for "
                  "image {0}".format(fname))

    # close all windows
    cv2.destroyAllWindows()

    # calibrate camera by computing camera matrix, distortion coefficients,
    # rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None, None)

    return ret, mtx, dist, rvecs, tvecs


def compute_camera_coefficients(fp: str = ""):
    """Computes the camera coefficients and saves them to a file."""

    # get location of this file
    HERE = os.path.dirname(sanitize_path(__file__))

    # sanitize input filepath
    if fp:
        if not os.path.isfile(fp):
            fp = sanitize_path(os.path.join(fp, "coefficients.yml"))
    else:
        fp = sanitize_path(os.path.join(HERE, "coefficients.yml"))

    # define data directories
    chessboard_dir = sanitize_path(os.path.join(HERE, "imgs_chessboard"))

    # find calibration images
    chessboard_imgs = glob.glob(os.path.join(chessboard_dir, "*.jpg"))

    # calibrate
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(chessboard_imgs)

    # print results
    print("[OPENCV] Camera matrix : \n")
    print(mtx)
    print("[OPENCV] Distortion : \n")
    print(dist)
    print("[OPENCV] Rotation : \n")
    print(rvecs)
    print("[OPENCV] Translation : \n")
    print(tvecs)

    # save the coefficients
    save_coefficients(mtx, dist, fp)

    # print info
    print("[OPENCV] Camera coefficients successfully saved to file: \n")
    print("[OPENCV] " + fp)

    # return the coefficients
    return (mtx, dist)


def detect_contours_from_file(filepath: str,
                              thresh_binary: int,
                              thresh_area: float,
                              approx: int = 0,
                              invert: bool = False):
    # read image from filepath
    image = cv2.imread(os.path.normpath(filepath))

    return detect_contours_from_image(image,
                                      thresh_binary,
                                      thresh_area,
                                      approx,
                                      invert)


def detect_contours_from_image(image: np.ndarray,
                               thresh_binary: int,
                               thresh_area: float,
                               approx: int = 0,
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

    # determine chain approximation
    chain_approx = cv2.CHAIN_APPROX_NONE
    if approx <= 0:
        chain_approx = cv2.CHAIN_APPROX_NONE
    elif approx == 1:
        chain_approx = cv2.CHAIN_APPROX_SIMPLE
    elif approx == 2:
        chain_approx = cv2.CHAIN_APPROX_TC89_L1
    else:
        chain_approx = cv2.CHAIN_APPROX_TC89_KCOS

    # find the contours from the thresholded image
    # CHAIN_APPROX_NONE
    # CHAIN_APPROX_SIMPLE
    # CHAIN_APPROX_TC89_L1
    # CHAIN_APPROX_TC89_KCOS
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           chain_approx)

    # sort contours by area and filter against threshold
    # only return outermost contours, no inner contours based on hierarchy
    if thresh_area > 0:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = [contours[i] for i in range(len(contours))
                    if areas[i] > thresh_area and
                    hierarchy[0][i][3] == -1]

    return image, contours


def read_image(filepath: str):
    """Wrapper for cv2.imread"""
    return cv2.imread(os.path.normpath(filepath))


def undistort_image(image, mtx, dist, remap: bool = True):
    # get height and width of input image
    h, w = image.shape[:2]
    # compute new camera matrix based on calibrated matrix and distortion
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                      dist,
                                                      (w, h),
                                                      1,
                                                      (w, h))
    # undistort using remap function
    if remap:
        mapx, mapy = cv2.initUndistortRectifyMap(mtx,
                                                 dist,
                                                 None,
                                                 newcameramtx,
                                                 (w, h),
                                                 5)
        undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # undistort using undistort function
    else:
        undistorted_image = cv2.undistort(image, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # return resulting undistorted image
    return undistorted_image


def warp_image(image, xform, width, height):
    """Wrapper for cv2.warpPerspective"""

    # pixels per inch
    ppi = 96
    # millimeters per inch
    mmpi = 25.4

    # dheight and dwidth in millimeters
    height = int((height / mmpi) * ppi)
    width = int((width / mmpi) * ppi)

    return cv2.warpPerspective(image, xform, (width, height))


# UTILITY FUNCTIONS -----------------------------------------------------------

def save_coefficients(mtx, dist, path):
    """
    Save the camera matrix and the distortion coefficients to given path/file.
    """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()


def load_coefficients(path):
    """Loads camera matrix and distortion coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return (camera_matrix, dist_matrix)


def reset_windows():
    """Wrapper for cv2.destroyAllWindows()"""
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()


# TEST MODULE -----------------------------------------------------------------

def test_detect_contours_from_file():
    # use the demo image to perform some contour detection
    thisfolder = os.path.dirname(sanitize_path((__file__)))
    fp = sanitize_path(thisfolder + r"\demo_image.jpg")
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


# MAIN ROUTINE ----------------------------------------------------------------

if __name__ == "__main__":
    # test contour detection
    # test_detect_contours()
    # capture_image()
    # test_detect_contours_from_image()

    # calibrate_camera(1)
    pass
