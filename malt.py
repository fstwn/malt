# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import clr
from itertools import product
from os.path import normpath

# OPTIONS ---------------------------------------------------------------------

# Set to True to run using Flask
_FLASK = False

# True if you want to run using Rhino.Inside.CPython
_RHINOINSIDE = True

# System directory of your Rhino installation (for Rhino.Inside.CPython)
_RHINODIR = r"C:\Program Files\Rhino 7\System"

# Set to True to enable System import
_USING_SYSTEM = False

# Set to True to enable Grasshopper import
_USING_GH = False

# Set to True to enable Kangaroo 2 import
_USING_K2 = False


# HOPS & RHINO SETUP ----------------------------------------------------------

import ghhops_server as hs # NOQA402

if _FLASK and _RHINOINSIDE:
    raise ValueError("Server cannot run using Rhino.Inside *and* Flask. If "
                     "you want to use Rhino.Inside, use a standard HTTP Hops "
                     "Server. If you want to run the Server using Flaks, you "
                     "have to use rhino3dm instead of Rhino.Inside "
                     "(deactivate the _RHINOINSIDE option to do so)!")

# RHINO.INSIDE OR RHINO3DM
if _RHINOINSIDE:
    import rhinoinside
    rhinoinside.load(rhino_dir=normpath(_RHINODIR))
    import Rhino # NOQA402
else:
    import rhino3dm # NOQA402

# SYSTEM IF NECESSARY
if _USING_SYSTEM:
    import System # NOQA402

# GRASSHOPPER IF NECESSARY
if _USING_GH:
    clr.AddReference("Grasshopper.dll")
    import Grasshopper as gh # NOQA402

# KANGAROO 2 IF NECESSARY
if _USING_K2:
    clr.AddReference("KangarooSolver.dll")
    import KangarooSolver as ks # NOQA402

# MODULE IMPORTS --------------------------------------------------------------

import numpy as np # NOQA402
import open3d as o3d # NOQA402
import igl # NOQA402


# LOCAL MODULE IMPORTS --------------------------------------------------------

import localmodules.hopsutilities as hsutil # NOQA402

from localmodules import icp # NOQA402
from localmodules import intri # NOQA402


# REGSISTER FLASK OR RHINOINSIDE HOPS APP -------------------------------------
if _FLASK:
    from flask import Flask # NOQA402
    flaskapp = Flask(__name__)
    hops = hs.Hops(app=flaskapp)
elif _RHINOINSIDE:
    hops = hs.Hops(app=rhinoinside)
else:
    hops = hs.Hops()

# HOPS COMPONENTS -------------------------------------------------------------

@hops.component(
    "/icp.RegisterPointClouds",
    name="RegisterPointClouds",
    nickname="Register",
    description="Register a Scene PointCloud with a given Model PointCloud.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("ScenePoints", "S", "ScenePoints to evaluate", hs.HopsParamAccess.LIST),
        hs.HopsPoint("ModelPoints", "M", "ModelPoints for ICP", hs.HopsParamAccess.LIST),
        hs.HopsNumber("Threshold", "T", "Threshold for convergence", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("MaxIterations", "I", "Maximum iterations", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("NNAlgorithm", "A", "Algorithm used for nearest neighbor computation, can be 0 (KNN) or 1 (Hungarian). Defaults to KNN.", hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsPoint("RegisteredPoints", "R", "Regsitered ScenePoints", hs.HopsParamAccess.LIST),
        hs.HopsNumber("Transform", "X", "Transformation Matrix", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Error", "E", "Mean Error of ICP operation", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Iterations", "I", "Iterations before termination.", hs.HopsParamAccess.ITEM)
    ])
def icp_RegsiterPointClouds(scene_pts,
                            model_pts,
                            threshold=1e-3,
                            max_iters=20,
                            alg=0):

    # sanitize alg input
    if alg == 0:
        alg = "knn"
    elif alg == 1:
        alg = "hungarian"
    else:
        raise RuntimeError("NNAlgorithm has to be either 0 or 1!")

    # convert points to lists
    scene_pt_list = [[pt.X, pt.Y, pt.Z] for pt in scene_pts]
    model_pt_list = [[pt.X, pt.Y, pt.Z] for pt in model_pts]

    # get iterative closest point result
    res = icp.repeat_icp_until_good_fit(scene_pt_list,
                                        model_pt_list,
                                        threshold,
                                        10,
                                        max_iterations=max_iters,
                                        tolerance=1e-3,
                                        nn_alg=alg)

    # convert the transformation array to an actual rhino transform
    xform = hsutil.np_array_to_rhino_transform(res[0])

    # copy scene points and transform the copy using the xform
    transformed_pts = scene_pts[:]
    [pt.Transform(xform) for pt in transformed_pts]

    xformlist = []
    for i, j in product(range(4), range(4)):
        xformlist.append(float(res[0][i][j]))
    err = float(res[1])
    iters = res[2]

    return transformed_pts, xformlist, err, iters


@hops.component(
    "/open3d.PoissonMesh",
    name="PoissonMesh",
    nickname="PoissonMesh",
    description="Construct a Mesh from a PointCloud using Open3D poisson mesh reconstruction.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST),
        # hs.HopsVector("Normals", "N", "Optional Normals of the pointcloud to be used in reconstruction", hs.HopsParamAccess.LIST, optional=True, default=None),
        hs.HopsInteger("Depth", "D", "Depth parameter for the poisson algorithm. Defaults to 8.", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Width", "W", "Width parameter for the poisson algorithm. Ignored if depth is specified. Defaults to 0.", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Scale", "S", "Scale parameter for the poisson algorithm.", hs.HopsParamAccess.ITEM),
        hs.HopsBoolean("LinearFit", "L", "If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.", hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM),
    ])
def open3d_PoissonMeshComponent(points,
                                depth=8,
                                width=0,
                                scale=1.1,
                                linear_fit=False):

    # convert point list to np array
    np_points = np.array([[pt.X, pt.Y, pt.Z] for pt in points])

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    # if normals:
    #     np_normals = np.array([[n.X, n.Y, n.Z] for n in normals])
    #     pointcloud.normals = o3d.utility.Vector3dVector(np_normals)
    # else:
    #     pointcloud.estimate_normals()
    # estimate normals of incoming points
    pointcloud.estimate_normals()

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pointcloud,
                        depth=depth,
                        width=width,
                        scale=scale,
                        linear_fit=linear_fit)[0]
    bbox = pointcloud.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # create rhino mesh from o3d output
    rhino_mesh = Rhino.Geometry.Mesh()
    [rhino_mesh.Vertices.Add(v[0], v[1], v[2])
     for v in np.asarray(p_mesh_crop.vertices)]
    [rhino_mesh.Faces.AddFace(f[0], f[1], f[2])
     for f in np.asarray(p_mesh_crop.triangles)]

    rhino_mesh.Normals.ComputeNormals()
    rhino_mesh.Compact()

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.PoissonMeshNormals",
    name="PoissonMeshNormals",
    nickname="PoissonMeshNormals",
    description="Construct a Mesh from a PointCloud and corresponding normals using Open3D poisson mesh reconstruction.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST),
        hs.HopsVector("Normals", "N", "The Normals of the pointcloud to be used in reconstruction", hs.HopsParamAccess.LIST),
        hs.HopsInteger("Depth", "D", "Depth parameter for the poisson algorithm. Defaults to 8.", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("Width", "W", "Width parameter for the poisson algorithm. Ignored if depth is specified. Defaults to 0.", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Scale", "S", "Scale parameter for the poisson algorithm.", hs.HopsParamAccess.ITEM),
        hs.HopsBoolean("LinearFit", "L", "If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.", hs.HopsParamAccess.ITEM)
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM),
    ])
def open3d_PoissonMeshNormalsComponent(points,
                                       normals,
                                       depth=8,
                                       width=0,
                                       scale=1.1,
                                       linear_fit=False):

    # convert point list to np array
    np_points = np.array([[pt.X, pt.Y, pt.Z] for pt in points])

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    if normals:
        np_normals = np.array([[n.X, n.Y, n.Z] for n in normals])
        pointcloud.normals = o3d.utility.Vector3dVector(np_normals)
    else:
        pointcloud.estimate_normals()

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pointcloud,
                        depth=depth,
                        width=width,
                        scale=scale,
                        linear_fit=linear_fit)[0]
    bbox = pointcloud.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # create rhino mesh from o3d output
    rhino_mesh = Rhino.Geometry.Mesh()
    [rhino_mesh.Vertices.Add(v[0], v[1], v[2])
     for v in np.asarray(p_mesh_crop.vertices)]
    [rhino_mesh.Faces.AddFace(f[0], f[1], f[2])
     for f in np.asarray(p_mesh_crop.triangles)]

    rhino_mesh.Normals.ComputeNormals()
    rhino_mesh.Compact()

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/intri.IntrinsicTriangulation",
    name="IntrinsicTriangulation",
    nickname="InTri",
    description="Compute intrinsic triangulation of a triangle mesh.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to create intrinsic triangulation for.", hs.HopsParamAccess.ITEM),
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh with an intrinsic triangulation.", hs.HopsParamAccess.ITEM),
    ])
def intri_IntrinsicTriangulationComponent(mesh):

    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # initialize the glue map and edge lengths arrays from the input data
    G = intri.build_gluing_map(F)
    eL = intri.build_edge_lengths(V, F)

    # flip to delaunay
    intri.flip_to_delaunay(F, G, eL)

    intri_mesh = mesh.Duplicate()
    intri_mesh.Faces.Clear()

    [intri_mesh.Faces.AddFace(face[0], face[1], face[2]) for face in F]
    intri_mesh.Normals.ComputeNormals()
    intri_mesh.Compact()

    return intri_mesh


@hops.component(
    "/intri.HeatMethodDistance",
    name="HeatMethodDistance",
    nickname="HeatDist",
    description="Compute geodesic distances using the heat method.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute geodesic distances on.", hs.HopsParamAccess.ITEM),
        hs.HopsInteger("SourceVertex", "S", "The index of the source vertex from which to compute geodesic distances using the heat method. Defaults to 0.", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("TextureScale", "T", "The texturescale used for setting the normalized values to the texture coordinates.", hs.HopsParamAccess.ITEM),
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The mesh with texture coordinates set to the normalized values.", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Distances", "D", "The geodesic distances to all mesh vertices.", hs.HopsParamAccess.LIST),
        hs.HopsNumber("Values", "V", "The normalized values for every mesh vertex.", hs.HopsParamAccess.LIST),
    ])
def intri_HeatMethodDistanceComponent(mesh,
                                      source_vertex=0,
                                      texture_scale=1.0):

    assert source_vertex <= mesh.Vertices.Count - 1, ("The index of the "
                                                      "source vertex cannot "
                                                      "exceed the vertex "
                                                      "count!")

    # get vertex and face array
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # initialize the edge lengths array from the input data
    eL = intri.build_edge_lengths(V, F)

    # compute geodesic heat distances
    geodists = [float(d) for d in
                intri.heat_method_distance_from_vertex(F, eL, source_vertex)]

    # normalize all values for setting texture coordinates
    vmin = min(geodists)
    vmax = max(geodists)
    mult = 1.0 / (vmax - vmin)
    values = [mult * (v - vmin) for v in geodists]

    # set texture coordinates of output mesh
    out_mesh = mesh.Duplicate()
    for i in range(out_mesh.Vertices.Count):
        out_mesh.TextureCoordinates.SetTextureCoordinate(
                    i,
                    Rhino.Geometry.Point2f(0.0, values[i] * texture_scale))

    return out_mesh, geodists, values


# RUN HOPS APP AS EITHER FLASK OR DEFAULT -------------------------------------

if __name__ == "__main__":
    print("-------------------------------------------------")
    print("Available Hops Components on this Server:\n")
    [print("{0} -> {1}".format(c, hops._components[c].description))
        for c in hops._components]
    print("-------------------------------------------------")

    if type(hops) == hs.HopsFlask:
        flaskapp.run()
    else:
        hops.start(debug=True)
