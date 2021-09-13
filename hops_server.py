# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import clr
from itertools import product

# OPTIONS ---------------------------------------------------------------------

# Set to True to run in debug mode.
_DEBUG = True

# Set to True to allow access via local network (only works with Flask app!)
# WARNING: THIS MIGHT BE A SECURITY RISK BECAUSE IT POTENTIALLY ALLOWS PEOPLE
# TO EXECUTE CODE ON YOUR MACHINE! ONLY USE THIS IN A TRUSTED NETWORK!
_NETWORK_ACCESS = False

# Set to True to run using Flask as middleware
_FLASK = True

# True if you want to run using Rhino.Inside.CPython
_RHINOINSIDE = True

# Set to True to enable System import
_USING_SYSTEM = True

# Set to True to enable Grasshopper import
_USING_GH = False

# Set to True to enable Kangaroo 2 import
_USING_K2 = False


# HOPS & RHINO SETUP ----------------------------------------------------------

import ghhops_server as hs # NOQA402


# Define a custom Hops Middleware to enable Rhino.Inside.CPython in
# combination with a Flask app (otherwise not possible)
class CustomHops(hs.Hops):
    """Custom Hops Middleware allowing Flask app to also run Rhino.Inside"""

    def __new__(cls,
                app=None,
                debug=False,
                force_rhinoinside=False,
                *args,
                **kwargs) -> hs.base.HopsBase:
        # set logger level
        hs.hlogger.setLevel(hs.logging.DEBUG if debug else hs.logging.INFO)

        # determine the correct middleware base on the source app being wrapped
        # when running standalone with no source apps
        if app is None:
            hs.hlogger.debug("Using Hops default http server")
            hs.params._init_rhino3dm()
            return hs.middlewares.HopsDefault()

        # if wrapping another app
        app_type = repr(app)
        # if app is Flask
        if app_type.startswith("<Flask"):
            if force_rhinoinside:
                hs.hlogger.debug("Using Hops Flask middleware and rhinoinside")
                hs.params._init_rhinoinside()
            else:
                hs.hlogger.debug("Using Hops Flask middleware and rhino3dm")
                hs.params._init_rhino3dm()
            return hs.middlewares.HopsFlask(app, *args, **kwargs)

        # if wrapping rhinoinside
        elif app_type.startswith("<module 'rhinoinside'"):
            # determine if running with rhino.inside.cpython
            # and init the param module accordingly
            if not CustomHops.is_inside():
                raise Exception("rhinoinside is not loaded yet")
            hs.hlogger.debug("Using Hops default http server with rhinoinside")
            hs.params._init_rhinoinside()
            return hs.middlewares.HopsDefault(*args, **kwargs)

        raise Exception("Unsupported app!")


print("-----------------------------------------------------")
print("[INFO] Hops Server Configuration:")
print("[INFO] SERVER:  {0}".format(
            "Flask App" if _FLASK else "Hops Default HTTP Server"))
print("[INFO] RHINO:   {0}".format(
            "Rhino.Inside.CPython" if _RHINOINSIDE else "rhino3dm"))
if _NETWORK_ACCESS:
    print("[INFO] NETWORK: \033[31mNetwork Access Enabled!")
    print("[WARNING] ENABLING NETWORK ACCESS MIGHT BE A SECURITY RISK \n"
          "BECAUSE IT POTENTIALLY ALLOWS PEOPLE TO EXECUTE CODE ON YOUR \n"
          "MACHINE! ONLY USE THIS IN A TRUSTED NETWORK!\033[0m")
else:
    print("[INFO] NETWORK: Localhost Only")
print("-----------------------------------------------------")

# RHINO.INSIDE OR RHINO3DM
if _RHINOINSIDE:
    print("\033[34m[INFO] Loading Rhino.Inside.CPython ...\033[0m")
    import rhinoinside
    rhinoinside.load()
    import Rhino # NOQA402
else:
    import rhino3dm # NOQA402

# SYSTEM IF NECESSARY
if _USING_SYSTEM:
    print("\033[34m[INFO] Loading System (.NET) ...\033[0m")
    import System # NOQA402

# GRASSHOPPER IF NECESSARY
if _USING_GH:
    print("\033[32m[INFO] Loading Grasshopper ...\033[0m")
    clr.AddReference("Grasshopper.dll")
    import Grasshopper as gh # NOQA402

# KANGAROO 2 IF NECESSARY
if _USING_K2:
    print("\033[33m[INFO] Loading Kangaroo2 ...\033[0m")
    clr.AddReference("KangarooSolver.dll")
    import KangarooSolver as ks # NOQA402

# MODULE IMPORTS --------------------------------------------------------------

import numpy as np # NOQA402
import open3d as o3d # NOQA402
import igl # NOQA402


# LOCAL MODULE IMPORTS --------------------------------------------------------

import malt.hopsutilities as hsutil # NOQA402
from malt import icp # NOQA402
from malt import intri # NOQA402
from malt import imgprocessing # NOQA402

# REGSISTER FLASK AND/OR RHINOINSIDE HOPS APP ---------------------------------
if _FLASK:
    from flask import Flask # NOQA402
    flaskapp = Flask(__name__)
    hops = CustomHops(app=flaskapp, force_rhinoinside=_RHINOINSIDE)
elif not _FLASK and _RHINOINSIDE:
    hops = CustomHops(app=rhinoinside)
else:
    hops = CustomHops()


# HOPS COMPONENTS -------------------------------------------------------------

# GET ALL AVAILABLE COMPONENTS ////////////////////////////////////////////////

@hops.component(
    "/hops.Components",
    name="AvailableComponents",
    nickname="Components",
    description="List all routes of the available components",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[],
    outputs=[
        hs.HopsString("Components", "C", "All available Hops Components on this server.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsString("Description", "D", "The descriptions of the components", hs.HopsParamAccess.LIST), # NOQA501
    ])
def hops_AvailableComponentsComponent():
    comps = []
    descr = []
    for c in hops._components:
        comps.append(str(c))
        descr.append(hops._components[c].description)

    return comps, descr


# ITERATIVE CLOSEST POINT /////////////////////////////////////////////////////

@hops.component(
    "/icp.RegisterPointClouds",
    name="RegisterPointClouds",
    nickname="Register",
    description="Register a Scene PointCloud with a given Model PointCloud.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("ScenePoints", "S", "ScenePoints to evaluate", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsPoint("ModelPoints", "M", "ModelPoints for ICP", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Threshold", "T", "Threshold for convergence", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("MaxIterations", "I", "Maximum iterations", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("NNAlgorithm", "A", "Algorithm used for nearest neighbor computation, can be 0 (KNN) or 1 (Hungarian). Defaults to KNN.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsPoint("RegisteredPoints", "R", "Regsitered ScenePoints", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Transform", "X", "Transformation Matrix", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Error", "E", "Mean Error of ICP operation", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Iterations", "I", "Iterations before termination.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def icp_RegisterPointCloudsComponent(scene_pts,
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
    xform = hsutil.np_array_to_rhino_transform(res[0], Rhino)

    # copy scene points and transform the copy using the xform
    transformed_pts = scene_pts[:]
    [pt.Transform(xform) for pt in transformed_pts]

    # convert rhino transformation to list of floats for output
    xformlist = []
    for i, j in product(range(4), range(4)):
        xformlist.append(float(res[0][i][j]))
    err = float(res[1])
    iters = res[2]

    # return the results
    return transformed_pts, xformlist, err, iters


# LIBIGL //////////////////////////////////////////////////////////////////////

@hops.component(
    "/igl.MeshIsocurves",
    name="MeshIsocurves",
    nickname="MeshIsoC",
    description="Compute isocurves based on a function using libigl",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute isocurves on.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Values", "V", "The function to compute as a list of values at each vertex position of the mesh.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Count", "C", "Number of Isocurves", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsPoint("VertexPositions", "V", "Vertex positions of the isocurves", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("EdgePositions", "E", "Edge positions of the isocurves", hs.HopsParamAccess.LIST), # NOQA501
    ])
def igl_MeshIsocurvesComponent(mesh, values, count):
    # check if mesh is all triangles
    if mesh.Faces.QuadCount > 0:
        raise ValueError("Mesh has to be triangular!")

    if not len(values) == mesh.Vertices.Count:
        raise ValueError("List of function values does not correspond with "
                         "mesh vertices!")

    # get np arrays of vertices and faces
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    values = np.array(values)

    isoV, isoE = igl.isolines(V, F, values, count)

    isoV = hsutil.np_array_to_rhino_points(isoV, Rhino)
    evalues = []
    [evalues.extend([int(edge[0]), int(edge[1])]) for edge in isoE]

    return isoV, evalues


# INTRINSIC TRIANGULATIONS ////////////////////////////////////////////////////

@hops.component(
    "/intri.IntrinsicTriangulation",
    name="IntrinsicTriangulation",
    nickname="InTri",
    description="Compute intrinsic triangulation of a triangle mesh.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to create intrinsic triangulation for.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh with an intrinsic triangulation.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def intri_IntrinsicTriangulationComponent(mesh):

    # get vertices and faces as numpy arrays
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # initialize the glue map and edge lengths arrays from the input data
    G = intri.build_gluing_map(F)
    eL = intri.build_edge_lengths(V, F)

    # flip to delaunay
    intri.flip_to_delaunay(F, G, eL)

    # duplicate original mesh and clear the faces
    intri_mesh = mesh.Duplicate()
    intri_mesh.Faces.Clear()

    # set the intrinsic traignulation as faces
    [intri_mesh.Faces.AddFace(face[0], face[1], face[2]) for face in F]

    # compute normals and compact
    intri_mesh.Normals.ComputeNormals()
    intri_mesh.Compact()

    # return results
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
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute geodesic distances on.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("SourceVertex", "S", "The index of the source vertex from which to compute geodesic distances using the heat method. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("TextureScale", "T", "The texturescale used for setting the normalized values to the texture coordinates.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The mesh with texture coordinates set to the normalized values.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Distances", "D", "The geodesic distances to all mesh vertices.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Values", "V", "The normalized values for every mesh vertex.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def intri_HeatMethodDistanceComponent(mesh,
                                      source_vertex=0,
                                      texture_scale=1.0):

    # ensure that the user picked a feasible index
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

    # return results
    return out_mesh, geodists, values


# OPEN3D //////////////////////////////////////////////////////////////////////

@hops.component(
    "/open3d.PoissonMesh",
    name="PoissonMesh",
    nickname="PoissonMesh",
    description="Construct a Mesh from a PointCloud using Open3D poisson mesh reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Depth", "D", "Depth parameter for the poisson algorithm. Defaults to 8.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "Width parameter for the poisson algorithm. Ignored if depth is specified. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Scale", "S", "Scale parameter for the poisson algorithm.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("LinearFit", "L", "If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
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

    # estimate the normals
    pointcloud.estimate_normals()

    # create poisson mesh reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pointcloud,
                        depth=depth,
                        width=width,
                        scale=scale,
                        linear_fit=linear_fit)[0]

    # get bbx and crop poisson mesh with bbx
    bbox = pointcloud.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # create rhino mesh from o3d output and add vertices and faces
    rhino_mesh = Rhino.Geometry.Mesh()
    [rhino_mesh.Vertices.Add(v[0], v[1], v[2])
     for v in np.asarray(p_mesh_crop.vertices)]
    [rhino_mesh.Faces.AddFace(f[0], f[1], f[2])
     for f in np.asarray(p_mesh_crop.triangles)]

    # compute normals and compact
    rhino_mesh.Normals.ComputeNormals()
    rhino_mesh.Compact()

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.PoissonMeshNormals",
    name="PoissonMeshNormals",
    nickname="PoissonMeshNormals",
    description="Construct a Mesh from a PointCloud and corresponding normals using Open3D poisson mesh reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsVector("Normals", "N", "The Normals of the pointcloud to be used in reconstruction", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Depth", "D", "Depth parameter for the poisson algorithm. Defaults to 8.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "Width parameter for the poisson algorithm. Ignored if depth is specified. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Scale", "S", "Scale parameter for the poisson algorithm.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("LinearFit", "L", "If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def open3d_PoissonMeshNormalsComponent(points,
                                       normals,
                                       depth=8,
                                       width=0,
                                       scale=1.1,
                                       linear_fit=False):

    # convert point list to np array
    np_points = hsutil.rhino_points_to_np_array(points)

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    # if normals are present, use them, otherwise estimate using subroutine
    if normals:
        np_normals = np.array([[n.X, n.Y, n.Z] for n in normals])
        pointcloud.normals = o3d.utility.Vector3dVector(np_normals)
    else:
        pointcloud.estimate_normals()

    # create poisson mesh reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pointcloud,
                        depth=depth,
                        width=width,
                        scale=scale,
                        linear_fit=linear_fit)[0]

    # get bbx and crop poisson mesh with bbx
    bbox = pointcloud.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # create rhino mesh from o3d output and vertices and faces
    rhino_mesh = Rhino.Geometry.Mesh()
    [rhino_mesh.Vertices.Add(v[0], v[1], v[2])
     for v in np.asarray(p_mesh_crop.vertices)]
    [rhino_mesh.Faces.AddFace(f[0], f[1], f[2])
     for f in np.asarray(p_mesh_crop.triangles)]

    # compute normals and compact
    rhino_mesh.Normals.ComputeNormals()
    rhino_mesh.Compact()

    # return the rhino mesh
    return rhino_mesh


# OPENCV //////////////////////////////////////////////////////////////////////

@hops.component(
    "/opencv.DetectContours",
    name="DetectContours",
    nickname="DetCon",
    description="Detect contours in an image using OpenCV.",
    category=None,
    subcategory=None,
    icon=None,
    inputs=[
        hs.HopsString("FilePath", "F", "The filepath of the image.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("BinaryThreshold", "B", "The threshold for binary (black & white) conversion of the image. Defaults to 170.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("AreaThreshold", "T", "The area threshold for filtering the returned contours in pixels. Defaults to 100.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsCurve("Contours", "C", "The detected contours as Polylines.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def opencv_DetectContoursComponent(filepath,
                                   bthresh=170,
                                   athresh=100.0):
    # run contour detection using opencv
    image, contours = imgprocessing.detect_contours(filepath,
                                                    bthresh,
                                                    athresh)
    # construct polylines from contour output
    plcs = []
    for cnt in contours:
        # crate .NET list because Polyline constructor won't correctly handle
        # python lists (it took a while to find that out....)
        if len(cnt) >= 2:
            ptlist = System.Collections.Generic.List[Rhino.Geometry.Point3d]()
            [ptlist.Add(Rhino.Geometry.Point3d(float(pt[0][0]),
                                               float(pt[0][1]),
                                               0.0)) for pt in cnt]
            ptlist.Add(Rhino.Geometry.Point3d(float(cnt[0][0][0]),
                                              float(cnt[0][0][1]),
                                              0.0))
            plcs.append(Rhino.Geometry.PolylineCurve(ptlist))

    # return output
    return plcs


# RUN HOPS APP AS EITHER FLASK OR DEFAULT -------------------------------------

if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("[INFO] Available Hops Components on this Server:")
    [print("\033[36m{0}\033[0m -> {1}".format(
        c, hops._components[c].description))
        for c in hops._components]
    print("-----------------------------------------------------")

    if type(hops) == hs.HopsFlask:
        if _NETWORK_ACCESS:
            flaskapp.run(debug=_DEBUG, host="0.0.0.0")
        else:
            flaskapp.run(debug=_DEBUG)
    else:
        hops.start(debug=_DEBUG)
