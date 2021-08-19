# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from itertools import product

# HOPS & RHINO IMPORTS --------------------------------------------------------

import ghhops_server as hs

import rhinoinside
rhinoinside.load(rhino_dir=r"C:\Program Files\Rhino 7\System")
import Rhino # NOQA402

# SYSTEM IF NECESSARY
# import System

# GRASSHOPPER IF NECESSARY
# clr.AddReference("Grasshopper.dll")
# import Grasshopper

# KANGAROO 2 IF NECESSARY
# clr.AddReference("KangarooSolver.dll")
# import KangarooSolver as ks

# MODULE IMPORTS --------------------------------------------------------------

import numpy as np # NOQA402
import open3d as o3d # NOQA402
import igl # NOQA402


# LOCAL MODULE IMPORTS --------------------------------------------------------

import localmodules.hopsutilities as hsutil # NOQA402

from localmodules import icp # NOQA402
from localmodules import intri # NOQA402


# REGSISTER FLASK OR RHINOINSIDE HOPS APP -------------------------------------
# flaskapp = Flask(__name__)
hops = hs.Hops(app=rhinoinside)


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
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM),
    ])
def open3d_PoissonMeshComponent(points):

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
                        depth=8,
                        width=0,
                        scale=1.1,
                        linear_fit=False)[0]
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
        hs.HopsVector("Normals", "N", "Optional Normals of the pointcloud to be used in reconstruction", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM),
    ])
def open3d_PoissonMeshNormalsComponent(points, normals):

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
                        depth=8,
                        width=0,
                        scale=1.1,
                        linear_fit=False)[0]
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


# RUN HOPS APP AS EITHER FLASK OR DEFAULT -------------------------------------

if __name__ == "__main__":
    if type(hops) == hs.HopsFlask:
        flaskapp.run()
    else:
        hops.start(debug=True)
