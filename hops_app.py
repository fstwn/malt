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
from localmodules import icp # NOQA402
from localmodules import intri


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
def icpRegsiterPointClouds(scene_pts,
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
    xform = transform_from_list(res[0])

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
    name="PoissonMeshO3D",
    nickname="PoissonMesh",
    description="Construct a Mesh from a PointCloud using Open3D",
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
def open3dPoissonMeshComponent(points):

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

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/igl.GeodesicHeatDistances",
    name="GeodesicHeatDistancesIGL",
    nickname="GeoHeatDist",
    description="Get fast approimate geodesic distances using the heat method",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsMesh("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM),
    ])

def iglGeodesicHeatDistancesComponent(points, normals=None):
    return


# ADDITIONAL FUNCTIONALITY ----------------------------------------------------

def transform_from_list(data):
    xform = Rhino.Geometry.Transform(1.0)
    for i, j in product(range(4), range(4)):
        xform[i, j] = data[i][j]
    return xform


# RUN FLASK APP ---------------------------------------------------------------

if __name__ == "__main__":
    if type(hops) == hs.HopsFlask:
        flaskapp.run()
    else:
        hops.start(debug=True)
