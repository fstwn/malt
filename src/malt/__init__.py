# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from __future__ import (absolute_import, division, print_function)
import os

from .__version__ import (__author__, __author_email__, __copyright__,
                          __description__, __license__, __title__, __url__,
                          __version__)


# PACKAGE MODULE IMPORTS ------------------------------------------------------

import malt.hopsutilities as hopsutilities # NOQA402
import malt.icp as icp # NOQA402
import malt.imgprocessing as imgprocessing # NOQA402
import malt.intri as intri # NOQA402
import malt.miphopper as miphopper # NOQA402
import malt.shapesph as shapesph # NOQA402
import malt.sshd as sshd # NOQA402
import malt.ft20 as ft20 # NOQA402


# DEFINITIONS -----------------------------------------------------------------

ROOTDIR = os.path.dirname(__file__)
"""str: Path to the root folder of the malt package."""

REPODIR = hopsutilities.sanitize_path(os.path.join(ROOTDIR, "../.."))
"""str: Path to the root folder of the malt repository."""

DATADIR = hopsutilities.sanitize_path(os.path.join(ROOTDIR, "../../data"))
"""str: Path to the data folder of the malt repository."""

TESTDIR = hopsutilities.sanitize_path(os.path.join(ROOTDIR, "../../tests"))
"""str: Path to the tests folder of the malt repository."""

IMGDIR = hopsutilities.sanitize_path(os.path.join(ROOTDIR, "imgprocessing"))
"""str: Path to the imgprocessing folder of the malt repository."""

__all__ = [
    "__author__", "__author_email__", "__copyright__", "__description__",
    "__license__", "__title__", "__url__", "__version__",
]
