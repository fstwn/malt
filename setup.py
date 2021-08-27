# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from __future__ import absolute_import, print_function
import io
from glob import glob
from os.path import abspath, basename, dirname, join, splitext

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

from setuptools import find_packages, setup

# REQUIREMENTS CHECKING -------------------------------------------------------

here = abspath(dirname(__file__))


def read(*names, **kwargs):
    return io.open(
        join(here, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


requirements = read('requirements.txt').split('\n')

keywords_list = []

about = {}
exec(read('src', 'malt', '__version__.py'), about)

# RUN SETUP -------------------------------------------------------------------

setup(
    name=about['__title__'],
    version=about['__version__'],
    license=about['__license__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords=keywords_list,
    install_requires=requirements,
    extras_require={},
    entry_points={},
)
