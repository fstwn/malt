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

from setuptools import (find_packages,
                        setup)

# REQUIREMENTS CHECKING -------------------------------------------------------

HERE = abspath(dirname(__file__))


def read(*names, **kwargs):
    return io.open(join(HERE, *names),
                   encoding=kwargs.get("encoding", "utf8")).read()


long_description = read("README.md")
requirements = [r for r in read("requirements.txt").split("\n") if r and not
                r.startswith("#")]

keywords_list = ["architecture", "engineering", "fabrication", "computation",
                 "geometry", "design", "Hops", "Grasshopper"]

about = {}
exec(read("src", "malt", "__version__.py"), about)


# RUN SETUP -------------------------------------------------------------------

setup(
    name=about["__title__"],
    version=about["__version__"],
    license=about["__license__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=keywords_list,
    install_requires=requirements,
    extras_require={},
    entry_points={},
)


# DEFINE FUNC FOR REPLACING  CODE IN PARAMS.PY ---- !!! DANGER ZONE !!! -------

def _fix_from_result_func(params_file):
    """Replace some source code in params.py to support datatree output."""
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
    # open file and read source code
    params_source = None
    with open(params_file, mode="r") as f:
        params_source = f.read()
    # if the original source code is in the file, replace it with the
    # augmented version for outputting trees
    if replacement_func_src in params_source:
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py already replaced before!")
    elif (original_func_src in params_source and
          replacement_func_src not in params_source):
        params_source = params_source.replace(original_func_src,
                                              replacement_func_src)
        with open(params_file, mode="w") as f:
            f.write(params_source)
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py was successfully replaced!")
    else:
        raise RuntimeError(
            "[ERROR] MALT post installation steps not completed! \n"
            "[ERROR] Source not found in params.py file!")


def _add_circle_param(params_file):
    """Replace some source code and add support for Circles as params"""
    # define original function source code to look for
    deactivated_key = ('    # "HopsCircle",')
    original_func_src = ("""
class HopsBrep(_GHParam):
    \"\"\"Wrapper for GH Brep\"\"\"

    param_type = \"Brep\"
    result_type = \"Rhino.Geometry.Brep"


class HopsCurve(_GHParam):""")
    # this is the fixed replacement source code
    activated_key = ('    "HopsCircle",')
    replacement_func_src = ("""
class HopsBrep(_GHParam):
    \"\"\"Wrapper for GH Brep\"\"\"

    param_type = "Brep"
    result_type = "Rhino.Geometry.Brep"


class HopsCircle(_GHParam):
    \"\"\"Wrapper for GH_Circle\"\"\"

    param_type = "Circle"
    result_type = "Rhino.Geometry.Circle"

    coercers = {
        "Rhino.Geometry.Circle": lambda d: HopsCircle._make_circle(
            HopsPlane._make_plane(d["Plane"]["Origin"],
                                  d["Plane"]["XAxis"],
                                  d["Plane"]["YAxis"]),
            d["Radius"]
        )
    }

    @staticmethod
    def _make_circle(p, r):
        if RHINO_GEOM.__name__ == "rhino3dm":
            raise NotImplementedError("Can't create plane-aligned circle "
                                      "using rhino3dm due to missing "
                                      "implementation!")
        return RHINO_GEOM.Circle(p, r)


class HopsCurve(_GHParam):""")
    # open file and read source code
    params_source = None
    with open(params_file, mode="r") as f:
        params_source = f.read()
    # if the original source code is in the file, replace it with the
    # augmented version for outputting trees
    if replacement_func_src in params_source:
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py already replaced before!")
    elif (original_func_src in params_source and
          replacement_func_src not in params_source):
        params_source = params_source.replace(deactivated_key,
                                              activated_key)
        params_source = params_source.replace(original_func_src,
                                              replacement_func_src)
        with open(params_file, mode="w") as f:
            f.write(params_source)
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py was successfully replaced!")
    else:
        raise RuntimeError(
            "[ERROR] MALT post installation steps not completed! \n"
            "[ERROR] Source not found in params.py file!")


def _add_plane_param(params_file):
    """Replace some source code and add support for Planes as params"""
    # define original function source code to look for
    deactivated_key = ('    # "HopsPlane",')
    original_func_src = ("""
class HopsNumber(_GHParam):
    \"\"\"Wrapper for GH Number\"\"\"

    param_type = "Number"
    result_type = "System.Double"

    coercers = {
        "System.Double": lambda d: float(d),
    }


class HopsPoint(_GHParam):""")
    # this is the fixed replacement source code
    activated_key = ('    "HopsPlane",')
    replacement_func_src = ("""
class HopsNumber(_GHParam):
    \"\"\"Wrapper for GH Number\"\"\"

    param_type = "Number"
    result_type = "System.Double"

    coercers = {
        "System.Double": lambda d: float(d),
    }


class HopsPlane(_GHParam):
    \"\"\"Wrapper for GH_Plane\"\"\"

    param_type ="Plane"
    result_type = "Rhino.Geometry.Plane"

    coercers = {
        "Rhino.Geometry.Plane": lambda p: HopsPlane._make_plane(p["Origin"],
                                                                p["XAxis"],
                                                                p["YAxis"])
    }

    @staticmethod
    def _make_plane(o, x, y):
        return RHINO_GEOM.Plane(RHINO_GEOM.Point3d(o["X"], o["Y"], o["Z"]),
                                RHINO_GEOM.Vector3d(x["X"], x["Y"], x["Z"]),
                                RHINO_GEOM.Vector3d(y["X"], y["Y"], y["Z"]))


class HopsPoint(_GHParam):""")
    # open file and read source code
    params_source = None
    with open(params_file, mode="r") as f:
        params_source = f.read()
    # if the original source code is in the file, replace it with the
    # augmented version for outputting trees
    if replacement_func_src in params_source:
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py already replaced before!")
    elif (original_func_src in params_source and
          replacement_func_src not in params_source):
        params_source = params_source.replace(deactivated_key,
                                              activated_key)
        params_source = params_source.replace(original_func_src,
                                              replacement_func_src)
        with open(params_file, mode="w") as f:
            f.write(params_source)
        print("[INFO] MALT post installation steps done!")
        print("[INFO] Source code in params.py was successfully replaced!")
    else:
        raise RuntimeError(
            "[ERROR] MALT post installation steps not completed! \n"
            "[ERROR] Source not found in params.py file!")


def run_post_install_steps():
    """Run all post install steps"""
    # import ghhops_server here
    try:
        import ghhops_server
    except ImportError:
        raise RuntimeError("[ERROR] ghhops_server could not be imported. If"
                           "ghhops_server is not installed, the component "
                           "server of MALT can't run!")
    # get ghhops server path
    ghh_path = abspath(ghhops_server.__file__)
    # get params.py file
    params_file = normpath("\\".join(ghh_path.split("\\")[:-1] +
                                     ["params.py"]))
    print("[INFO] params.py file: " + str(params_file))
    if (isfile(params_file) and ghhops_server.__version__ == "1.4.1"):
        # run subroutine post install steps
        _fix_from_result_func(params_file)
        _add_circle_param(params_file)
        _add_plane_param(params_file)
    else:
        if ghhops_server.__version__ != "1.4.1":
            raise RuntimeError(
                "[ERROR] ghhops_server version conflict! Has to be 1.4.1! \n"
                "[ERROR] MALT post Installation Steps could not be "
                "completed!")
        raise RuntimeError("[ERROR] ghhops_server/params.py could not be "
                           "found!")


# RUN POST INSTALLATION STEPS -------------------------------------------------

run_post_install_steps()
