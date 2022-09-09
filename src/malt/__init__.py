# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from __future__ import (absolute_import, division, print_function)
import os

from .__version__ import (__author__, __author_email__, __copyright__,
                          __description__, __license__, __title__, __url__,
                          __version__)


# PACKAGE MODULE IMPORTS ------------------------------------------------------

import malt.ghgurobi as ghgurobi # NOQA402
import malt.hopsutilities as hopsutilities # NOQA402
import malt.icp as icp # NOQA402
import malt.imgprocessing as imgprocessing # NOQA402
import malt.intri as intri # NOQA402
import malt.shapesph as shapesph # NOQA402
import malt.sshd as sshd # NOQA402


# DEFINITIONS -----------------------------------------------------------------

def sanitize(path):
    return os.path.normpath(os.path.abspath(path))


ROOTDIR = os.path.dirname(__file__)
"""str: Path to the root folder of the malt package."""

REPODIR = sanitize(os.path.join(ROOTDIR, "../.."))
"""str: Path to the root folder of the malt repository."""

DATADIR = sanitize(os.path.join(ROOTDIR, "../../data"))
"""str: Path to the data folder of the malt repository."""

TESTDIR = sanitize(os.path.join(ROOTDIR, "../../tests"))
"""str: Path to the tests folder of the malt repository."""


__all__ = [
    "__author__", "__author_email__", "__copyright__", "__description__",
    "__license__", "__title__", "__url__", "__version__",
]
