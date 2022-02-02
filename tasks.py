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
        self.out.write(message + '\n')
        self.out.flush()

    def info(self, message):
        self.write('[INFO] %s' % message)

    def warn(self, message):
        self.write('[WARN] %s' % message)


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
    with chdir(malt.TESTDIR):
        c.run("pytest")


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
