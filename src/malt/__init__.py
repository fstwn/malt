from __future__ import absolute_import, division, print_function
import sys

from .__version__ import (__author__, __author_email__, __copyright__,
                          __description__, __license__, __title__, __url__,
                          __version__)


import malt.icp as icp # NOQA402
import malt.imgprocessing as imgprocessing # NOQA402
import malt.intri as intri # NOQA402
# import tf_demo as tf_demo # NOQA402

if sys.platform == "win32":
    import malt.hopsutilities as hopsutilities # NOQA402


__all__ = [
    '__author__', '__author_email__', '__copyright__', '__description__',
    '__license__', '__title__', '__url__', '__version__',
]
