# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from itertools import product


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np


# RHINOINSIDE -----------------------------------------------------------------

import rhinoinside
rhinoinside.load(rhino_dir=r"C:\Program Files\Rhino 7\System")
import Rhino # NOQA402


# NUMPY & RHINO ---------------------------------------------------------------

def rhino_mesh_to_np_arrays(mesh: Rhino.Geometry.Mesh):
    """
    Converts a Rhino.Geometry.Mesh to numpy arrays of vertices and faces.
    """
    vertices = mesh.Vertices
    faces = mesh.Faces
    # V = np.array([[v.X,
    #                vertices[i].Y,
    #                vertices[i].Z] for i in range(vertices.Count)])

    # F = np.array([[faces[i].A,
    #                faces[i].B,
    #                faces[i].C] for i in range(faces.Count)])

    V = np.array([[v.X,
                   v.Y,
                   v.Z] for _, v in enumerate(mesh.Vertices)])

    F = np.array([[f.A,
                   f.B,
                   f.C] for _, f in enumerate(mesh.Faces)])

    return V, F


def np_array_to_rhino_transform(xform_np_array: np.array):
    """
    Converts a 4x4 numpy array to a Rhino.Geometry.Transform transformation
    matrix.
    """
    rhino_xform = Rhino.Geometry.Transform(1.0)
    for i, j in product(range(4), range(4)):
        rhino_xform[i, j] = xform_np_array[i][j]
    return rhino_xform
