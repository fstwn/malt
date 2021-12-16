# Interface to the executables of the reference implementation of the paper
# "Rotation Invariant Spherical Harmonic Representation of 3D Shape
# Descriptors" by Michael Kazhdan, Thomas Funkhouser, and Szymon Rusinkiewicz
# Eurographics Symposium on Geometry Processing (2003)
#
# For reference see:
# https://www.cs.jhu.edu/~misha/MyPapers/SGP03.pdf
# https://github.com/mkazhdan/ShapeSPH
#
# licensed under MIT License
# Python interface written by Max Eschenbach, DDU, TU-Darmstadt, 2021

# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os
import subprocess


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import numpy as np
from plyfile import (PlyData,
                     PlyElement)

# PATHS TO DIR AND EXECS ------------------------------------------------------

_HERE = os.path.dirname(__file__)
_TEMPFILE = os.path.normpath(os.path.join(_HERE, "in.ply"))
_RESULTFILE = os.path.normpath(os.path.join(_HERE, "result.txt"))
_SHAPEDESCRIPTOR_EXEC = os.path.normpath(os.path.join(_HERE,
                                                      "executables",
                                                      "ShapeDescriptor.exe"))


# FUNCTION DEFINITIONS --------------------------------------------------------

def _write_ply_tempfile(vertices: np.array, faces: np.array):
    """
    Write temporary .PLY file to execute shapedescriptor on...
    """
    el_v = PlyElement.describe(vertices, "vertex")
    el_f = PlyElement.describe(faces, "face")
    with open(_TEMPFILE, mode="wb") as f:
        PlyData([el_v, el_f], text=True).write(f)
    return _TEMPFILE


def _read_result():
    """
    Read the result file and return the values
    """
    lines = []
    with open(_RESULTFILE, "r") as f:
        lines = f.readlines()
    values = [float(x) for x in lines[1].split("\n")[0].split(" ")[1:]]
    return values


def compute_descriptor(vertices: np.array, faces: np.array):
    """
    Compute descriptor using executable
    """
    tempfile = _write_ply_tempfile(vertices, faces)
    command = (_SHAPEDESCRIPTOR_EXEC +
               " --in " + tempfile +
               " --out " + _RESULTFILE +
               " --verbose")
    return_code = subprocess.call(command)
    print(return_code)
    return _read_result()


# TESTING ---------------------------------------------------------------------

if __name__ == "__main__":
    command = (_SHAPEDESCRIPTOR_EXEC +
               " --in " + _TEMPFILE +
               " --out " + _RESULTFILE +
               " --verbose")
    return_code = subprocess.call(command)
    print(return_code)
    result = _read_result()
    print(result)
