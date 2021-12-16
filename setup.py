# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from __future__ import (absolute_import,
                        print_function)

import io
from glob import glob
from os.path import (abspath,
                     basename,
                     dirname,
                     isfile,
                     join,
                     normpath,
                     splitext)

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


# DEFINE FUNC FOR REPLACING  CODE IN PARAMS.PY ---- !!! DANGER ZONE !!! -------

def fix_from_result_func():
    """Replace some source code in params.py to support datatree output."""
    # import ghhops_server here
    try:
        import ghhops_server
    except ImportError:
        raise RuntimeError("WARNING! ghhops_server could not be imported. If"
                           "ghhops_server is not installed, the component "
                           "server of MALT can't run!")
    # define original function source code to look for
    original_func_src = ("""
    def from_result(self, value):
        \"\"\"Serialize parameter with given value for output\"\"\"
        if not isinstance(value, tuple) and not isinstance(value, list):
            value = (value,)

        output_list = [
            {
                "type": self.result_type,
                "data": RHINO_TOJSON(v)
            } for v in value
        ]

        output = {
            "ParamName": self.name,
            "InnerTree": {
                "0": output_list
            },
        }
        return output""")
    # this is the fixed replacement source code
    replacement_func_src = ("""
    def from_result(self, value):
        \"\"\"Serialize parameter with given value for output\"\"\"
        if self.access == HopsParamAccess.TREE and isinstance(value, dict):
            tree = {}
            for key in value.keys():
                branch_data = [
                    {
                        "type": self.result_type,
                        "data": RHINO_TOJSON(v)
                    } for v in value[key]
                ]
                tree[key] = branch_data
            output = {
                "ParamName": self.name,
                "InnerTree": tree,
            }
            return output

        if not isinstance(value, tuple) and not isinstance(value, list):
            value = (value,)

        output_list = [
            {
                "type": self.result_type,
                "data": RHINO_TOJSON(v)
            } for v in value
        ]

        output = {
            "ParamName": self.name,
            "InnerTree": {
                "0": output_list
            },
        }
        return output""")
    # get ghhops server path
    ghh_path = abspath(ghhops_server.__file__)
    # get params.py file
    params_file = normpath("\\".join(ghh_path.split("\\")[:-1] +
                                     ["params.py"]))
    print("[INFO] params.py file: " + str(params_file))
    # if file exists, open it and read source code
    if (isfile(params_file) and ghhops_server.__version__ == "1.4.1"):
        params_source = None
        with open(params_file, mode="r") as f:
            params_source = f.read()
        # if the original source code is in the file, replace it with the
        # augmented version for outputting trees
        if original_func_src in params_source:
            params_source = params_source.replace(original_func_src,
                                                  replacement_func_src)
            with open(params_file, mode="w") as f:
                f.write(params_source)
            print("[INFO] MALT post installation steps done!")
            print("[INFO] Source code in params.py was successfully replaced!")
        elif replacement_func_src in params_source:
            print("[INFO] MALT post installation steps done!")
            print("[INFO] Source code in params.py already replaced before!")
        else:
            print("[WARNING] MALT post installation steps not completed!")
            print("[WARNING] Source not found in params.py file!")
    else:
        if ghhops_server.__version__ != "1.4.1":
            print("[WARNING] ghhops_server version conflict! Has to be 1.4.1!")
        print("[WARNING] MALT post Installation Steps could not be completed!")
        print("[WARNING] ghhops_server/params.py could not be found!")


# RUN POST INSTALLATION STEPS -------------------------------------------------

fix_from_result_func()
