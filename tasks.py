# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import contextlib
import os
import sys


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

from invoke import task, exceptions


# LOCAL MODULE IMPORTS --------------------------------------------------------

import malt


# LOGGING ---------------------------------------------------------------------

class Log(object):
    def __init__(self, out=sys.stdout, err=sys.stderr):
        self.out = out
        self.err = err

    def flush(self):
        self.out.flush()
        self.err.flush()

    def write(self, message):
        self.flush()
        self.out.write("%s\n" % message)
        self.out.flush()

    def info(self, message):
        self.write("[INFO] %s" % message)

    def warn(self, message):
        self.write("[WARNING] %s" % message)


log = Log()


# TASK DEFINITIONS ------------------------------------------------------------

@task(default=True)
def help(c):
    """
    Lists all available tasks and info on their usage.
    """
    c.run("invoke --list")
    log.info("Use \"invoke -h <taskname>\" to get detailed help for a task.")


@task()
def lint(c):
    """
    Check the coding style using flake8 python linter.
    """
    with chdir(malt.REPODIR):
        log.info("Running flake8 python linter on source folder...")
        c.run("flake8 --statistics src")


@task()
def check(c):
    """
    Perform various checks such as linting, etc.
    """
    with chdir(malt.REPODIR):
        lint(c)
        log.info("All checks passed.")


@task(help={
    "checks": ("Set to True to run all checks before running tests. "
               "Defaults to False")})
def test(c, checks=False):
    """
    Run all tests.
    """
    if checks:
        check(c)

    log.info("Running all tests...")
    with chdir(malt.TESTDIR):
        c.run("coverage run -m pytest")
        log.info("Analyzing coverage....")
        c.run("coverage report -m")


@task()
def gource(c):
    """
    Create gource video in /viz folder.
    """
    repodir = malt.REPODIR
    with chdir(repodir):
        vizpath = os.path.join(repodir, "viz")
        if not os.path.exists(vizpath):
            os.makedirs(vizpath)

        # Gource visualization
        try:
            # overview
            log.info("Creating gource overview visualization...")
            c.run(("gource {0} -1920x1080 -f --multi-sampling -a 1 -s 1 "
                   "--hide bloom,mouse,progress --camera-mode overview -r 60 "
                   "-o viz/overview.ppm").format(repodir))
            # track
            log.info("Creating gource track visualization...")
            c.run(("gource s{0} -1920x1080 -f --multi-sampling -a 1 -s 1 "
                   "--hide bloom,mouse,progress --camera-mode track -r 60 -o "
                   "viz/track.ppm").format(repodir))
        except exceptions.UnexpectedExit:
            log.warn("Gource is not installed or not in the current PATH! "
                     "See https://gource.io/ for info on installation.")

        # FFmpeg conversion
        try:
            log.info("Converting using FFMPEG...")
            c.run("ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i "
                  "viz/overview.ppm -vcodec libx264 -preset medium "
                  "-pix_fmt yuv420p -crf 1 -threads 0 -bf 0 viz/overview.mp4")
            os.remove("viz/overview.ppm")
            c.run("ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i "
                  "viz/track.ppm -vcodec libx264 -preset medium "
                  "-pix_fmt yuv420p -crf 1 -threads 0 -bf 0 viz/track.mp4")
            os.remove("viz/track.ppm")
        except exceptions.UnexpectedExit:
            log.warn("FFmpeg is not installed or not in the current PATH! "
                     "See https://ffmpeg.org/ for info on installation.")


@task()
def imgcalibration(c):
    """
    Run image camera calibration routine from imgprocessing module
    """

    with chdir(malt.IMGDIR):
        log.info("Running camera calibration routine...")
        malt.imgprocessing.compute_camera_coefficients()


@task()
def imgundistortion(c):
    """
    Run image undistortion routine from imgprocessing module.
    """

    with chdir(malt.IMGDIR):
        log.info("Running undistortion routine...")
        malt.imgprocessing.undistort_image_files()


@task(help={
    "w": ("Width of the working area."),
    "h": ("Height of the working area.")})
def imgperspective(c, w=1131, h=1131):
    """
    Run image perspective calibration and save transformation matrix to file.
    """

    with chdir(malt.IMGDIR):
        log.info("Running camera perspective calibration routine...")
        log.info("Width: {0} // Height: {1}".format(w, h))
        malt.imgprocessing.compute_perspective_xform(dwidth=w, dheight=h)


# CONTEXT ---------------------------------------------------------------------

@contextlib.contextmanager
def chdir(dirname=None):
    current_dir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(current_dir)
