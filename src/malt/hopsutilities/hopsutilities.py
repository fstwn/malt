# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from itertools import product


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np


# NUMPY & RHINO ---------------------------------------------------------------

def rhino_points_to_np_array(points):
    """
    Converts a list of Rhino.Geometry.Point3d objects to a #Nx3 numpy array.
    """
    return np.array([[pt.X, pt.Y, pt.Z] for pt in points])


def rhino_mesh_to_np_arrays(mesh):
    """
    Converts a Rhino.Geometry.Mesh to numpy arrays of vertices and faces.
    """
    V = np.array([[v.X, v.Y, v.Z] for _, v in enumerate(mesh.Vertices)])
    F = np.array([[f.A, f.B, f.C] for _, f in enumerate(mesh.Faces)])
    return V, F


def gh_tree_to_np_array(data_tree):
    """
    Converts a Grasshopper.DataTree to a numpy array. Returns a tuple of the
    paths and the actual array.
    """
    paths = list(data_tree.keys())
    np_data = np.array([data_tree[p] for p in paths])
    return (paths, np_data)


def np_array_to_rhino_transform(xform_np_array: np.array, Rhino=None):
    """
    Converts a 4x4 numpy array to a Rhino.Geometry.Transform transformation
    matrix.

    Remarks
    -------
    Caller needs to pass an imported Rhino to run this function. This is to
    avoid rhinoinside.load multiple times.
    """
    if not Rhino:
        raise ValueError("No Rhino instance supplied!")
    rhino_xform = Rhino.Geometry.Transform(1.0)
    for i, j in product(range(4), range(4)):
        rhino_xform[i, j] = xform_np_array[i][j]
    return rhino_xform


def np_array_to_rhino_points(pt_np_array: np.array, Rhino=None):
    """
    Converts a Nx3 numpy array to a list of Rhino.Geometry.Point3d objects.

    Remarks
    -------
    Caller needs to pass an imported Rhino to run this function. This is to
    avoid rhinoinside.load multiple times.
    """
    if not Rhino:
        raise ValueError("No Rhino instance supplied!")
    return [Rhino.Geometry.Point3d(float(v[0]), float(v[1]), float(v[2]))
            for v in pt_np_array]
