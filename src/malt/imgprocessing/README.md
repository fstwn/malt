# `malt` submodule for image processing using OpenCV

## Getting started with image processing and contour detection

### 1. Image directories

For convenience reasons, this directory contains a number of "magic"
directories to read and write image files during contour detection.

- `imgs_chessboard`: This is the directory for the chessboard calibration.
Place all images that you want to use for the chessboard calibration in this
folder as `.jpg` files!
- `imgs_raw`: This is the directory for the source images for undsitortion.
Place all images that you want to use for contour detection here (directly from
the camera) as `.jpg`files!
- `imgs_undistorted`: This is the directory where the undistortion routine will
write the undistorted images to. Do *NOT* place images here manually. The
undistorted output images will end up here automagically. After undistortion,
these are the files to perform contour detection on!

### 2. Computing the camera matrix and coefficients

The first preparation step for contour detection is computing the camera matrix
and coefficients using chessboard calibration. **Make sure all your images for
chessboard calibration are stored in the `imgs_chessboard` directory!**

For completeness reasons, start by `cd` into *YOUR* `malt` repository
directory. For me that's running
```
cd C:\source\repos\malt
```

If you don't have done so already, activate the conda environment
```
conda activate ddu_ias_research
```

Now you're ready! For performing the chessboard calibration you just need to
run
```
invoke imgcalibration
```

...and you should end up with something like this:
![Chessboard Calibration](../../../resources/readme/invoke_imgcalibration.png)