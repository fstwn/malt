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
        self.out.write("[STAT] %s" % message)
        self.out.flush()

    def info(self, message):
        self.write("[INFO] %s" % message)

    def warn(self, message):
        self.write("[WARN] %s" % message)


log = Log()


# TASK DEFINITIONS ------------------------------------------------------------

@task()
def lint(c):
    """
    Check the coding style using flake8 python linter.
    """
    log.write("Running flake8 python linter on source folder...")
    c.run("flake8 src")


@task()
def test(c):
    """
    Run all tests.
    """
    log.write("Running all tests...")
    with chdir(malt.TESTDIR):
        c.run("pytest")


@task()
def gource(c):
    """
    Create gource video.
    """
    log.write("Creating gource visualization...")
    with chdir(malt.REPODIR):
        c.run(("gource {0} -1920x1080 -f --multi-sampling -a 1 -s 1 "
               "--hide bloom,mouse,progress --camera-mode overview -r 60 -o "
               "a1_s1_overview.ppm").format(malt.REPODIR))
        log.write("Converting using FFMPEG...")
        c.run("ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i "
              "a1_s1_overview.ppm -vcodec libx264 -preset medium "
              "-pix_fmt yuv420p -crf 1 -threads 0 -bf 0 a1_s1_overview.mp4")
        os.remove("a1_s1_overview.ppm")


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
