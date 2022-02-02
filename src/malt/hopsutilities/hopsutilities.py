# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from itertools import product
from re import split as regexsplit
from typing import List


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np


# FUNCTION DEFINITIONS --------------------------------------------------------

# NUMPY & RHINO ///////////////////////////////////////////////////////////////

def rhino_points_to_np_array(points):
    """
    Converts a list of Rhino.Geometry.Point3d objects to a #Nx3 numpy array.
    """
    return np.array([[pt.X, pt.Y, pt.Z] for pt in points])


def rhino_mesh_to_np_arrays(mesh):
    """
    Converts a Rhino.Geometry.Mesh to numpy arrays of vertices and faces.
    """
    if mesh.Faces.QuadCount > 0:
        raise ValueError("Mesh has to be triangular!")
    V = np.array([[v.X, v.Y, v.Z] for _, v in enumerate(mesh.Vertices)])
    F = np.array([[f.A, f.B, f.C] for _, f in enumerate(mesh.Faces)])
    return V, F


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


def np_array_to_rhino_vectors(vec_np_array: np.array, Rhino=None):
    """
    Converts a Nx3 numpy array to a list of Rhino.Geometry.Vector3d objects.

    Remarks
    -------
    Caller needs to pass an imported Rhino to run this function. This is to
    avoid rhinoinside.load multiple times.
    """
    if not Rhino:
        raise ValueError("No Rhino instance supplied!")
    dims = vec_np_array.shape[1]
    assert dims == 2 or dims == 3, "Rhino Vectors can only be 2D or 3D!"
    if dims == 2:
        return [Rhino.Geometry.Vector3d(float(v[0]), float(v[1]), 0.0)
                for v in vec_np_array]
    else:
        return [Rhino.Geometry.Vector3d(float(v[0]), float(v[1]), float(v[2]))
                for v in vec_np_array]


# NUMPY & HOPS DATA ///////////////////////////////////////////////////////////

def hops_path_to_tuple(path: str):
    """
    Converts a hops data tree path to a python tuple.
    """
    return tuple(int(x) for x in regexsplit("{|;|}", path)[1:-1])


def hops_paths_to_tuples(paths: List[str]):
    """
    Converts a list of hops data tree paths to a list of python tuples.
    """
    return [tuple(int(x) for x in regexsplit("{|;|}", p)[1:-1]) for p in paths]


def hops_tree_to_np_array(data_tree: dict, tuplepaths: bool = False):
    """
    Converts a Hops DataTree (dict with paths as keys) to a numpy array.
    Returns a tuple of the paths as list and the actual array.
    """
    paths = list(data_tree.keys())
    np_data = np.array([data_tree[p] for p in paths])
    if tuplepaths:
        paths = hops_paths_to_tuples(paths)
    return (paths, np_data)


def hops_tree_verify(data_tree: dict):
    """
    Verifies the integrity of a Hops DataTree, ensuring that every path has
    the same shape.
    """
    tp = hops_paths_to_tuples(data_tree.keys())
    return True if all([len(tp[0]) == len(p) for p in tp]) else False


def np_float_array_to_hops_tree(np_array: np.array, paths: list = []):
    """
    Converts a numpy float array to a Hops DataTree (dict with paths as keys).
    """
    if not paths:
        paths = ["{0;" + str(x) + "}" for x in range(np_array.shape[0])]
    tree = {}
    if len(np_array.shape) == 1:
        for i, branch in enumerate(np_array):
            tree[paths[i].strip("}{")] = [float(branch)]
    elif len(np_array.shape) == 2:
        for i, branch in enumerate(np_array):
            tree[paths[i].strip("}{")] = [float(v) for v in branch]
    return tree


def np_int_array_to_hops_tree(np_array: np.array, paths: list = []):
    """
    Converts a numpy int array to a Hops DataTree (dict with paths as keys).
    """
    if not paths:
        paths = ["{0;" + str(x) + "}" for x in range(np_array.shape[0])]
    tree = {}
    if len(np_array.shape) == 1:
        for i, branch in enumerate(np_array):
            tree[paths[i].strip("}{")] = [int(branch)]
    elif len(np_array.shape) == 2:
        for i, branch in enumerate(np_array):
            tree[paths[i].strip("}{")] = [int(v) for v in branch]
    return tree


# RHINO & PLYFILE /////////////////////////////////////////////////////////////

def rhino_mesh_to_ply_elements(mesh):
    """
    Return vertex and face elements to write a plyfile
    """
    if mesh.Faces.QuadCount > 0:
        raise ValueError("Mesh has to be triangular!")
    V = np.array([(v.X, v.Y, v.Z) for _, v in enumerate(mesh.Vertices)],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    F = np.array([([f.A, f.B, f.C], 0, 0, 0)
                  for _, f in enumerate(mesh.Faces)],
                 dtype=[("vertex_indices", "i4", (3,)),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1")])
    return V, F
