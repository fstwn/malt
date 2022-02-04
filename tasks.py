# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import contextlib
import os
import sys


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

from invoke import task


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
        self.out.write("%s" % message)
        self.out.flush()

    def info(self, message):
        self.write("[INFO] %s" % message)

    def warn(self, message):
        self.write("[WARNING] %s" % message)


log = Log()


# TASK DEFINITIONS ------------------------------------------------------------

@task()
def lint(c):
    """
    Check the coding style using flake8 python linter.
    """
    log.info("Running flake8 python linter on source folder...")
    c.run("flake8 src")


@task()
def test(c):
    """
    Run all tests.
    """
    log.info("Running all tests...")
    with chdir(malt.TESTDIR):
        c.run("pytest")


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
        # overview
        log.info("Creating gource overview visualization...")
        c.run(("gource {0} -1920x1080 -f --multi-sampling -a 1 -s 1 "
               "--hide bloom,mouse,progress --camera-mode overview -r 60 -o "
               "viz/overview.ppm").format(repodir))
        log.info("Converting using FFMPEG...")
        c.run("ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i "
              "viz/overview.ppm -vcodec libx264 -preset medium "
              "-pix_fmt yuv420p -crf 1 -threads 0 -bf 0 viz/overview.mp4")
        os.remove("viz/overview.ppm")
        # track
        log.info("Creating gource track visualization...")
        c.run(("gource {0} -1920x1080 -f --multi-sampling -a 1 -s 1 --hide "
               "bloom,mouse,progress --camera-mode track -r 60 -o "
               "viz/track.ppm").format(repodir))
        log.info("Converting using FFMPEG...")
        c.run("ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i "
              "viz/track.ppm -vcodec libx264 -preset medium "
              "-pix_fmt yuv420p -crf 1 -threads 0 -bf 0 viz/track.mp4")
        os.remove("viz/track.ppm")


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
