# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import argparse
import clr
from itertools import product
import logging
import os


# COMMAND LINE ARGUMENT PARSING -----------------------------------------------

# Create argument parser
arg_parser = argparse.ArgumentParser(description="Process arguments for MALT "
                                                 "component server.")
# Create arguments
arg_parser.add_argument("-d", "--debug",
                        action="store_true",
                        required=False,
                        help="Activates Flask debug mode. "
                             "Defaults to False.",
                        dest="debug")
arg_parser.add_argument("-n", "--networkaccess",
                        action="store_true",
                        required=False,
                        help="Activates network access mode. "
                             "Defaults to False.",
                        dest="networkaccess")
arg_parser.add_argument("-f", "--noflask",
                        action="store_false",
                        required=False,
                        help="Runs server using Hops standard HTTP server. "
                             "Defaults to False (uses Flask as middleware).",
                        dest="flask")
# Parse all command line arguments
cl_args = arg_parser.parse_args()


# OPTIONS ---------------------------------------------------------------------

# Make matplotlib logger less verbose to prevent imports in
# referenced libraries from triggering a wall of debug messages.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Set to True to run in debug mode.
_DEBUG = True  # cl_args.debug

# Set to True to allow access via local network (only works with Flask app!)
# WARNING: THIS MIGHT BE A SECURITY RISK BECAUSE IT POTENTIALLY ALLOWS PEOPLE
# TO EXECUTE CODE ON YOUR MACHINE! ONLY USE THIS IN A TRUSTED NETWORK!
_NETWORK_ACCESS = cl_args.networkaccess

# Set to True to run using Flask as middleware
_FLASK = cl_args.flask

# True if you want to run using Rhino.Inside.CPython
_RHINOINSIDE = True

# Set to True to enable System import
_USING_SYSTEM = True

# Set to True to enable Grasshopper import
_USING_GH = False

# Set to True to enable Kangaroo2 import
_USING_K2 = False


# HOPS & RHINO SETUP ----------------------------------------------------------

import ghhops_server as hs # NOQA402


# Define a custom Hops class to enable Rhino.Inside.CPython in
# combination with a Flask app (otherwise not possible)
class ExtendedHops(hs.Hops):
    """
    Custom extended Hops class allowing Flask app to also run Rhino.Inside.
    """

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
            if not ExtendedHops.is_inside():
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
    print("[INFO] NETWORK: Network Access Enabled!")
    print("[WARNING] Enabling network access is a security risk because \n"
          "it potentially allows people to execute python code on your \n"
          "machine! Only use this option in a trusted network/environment!")
else:
    print("[INFO] NETWORK: Localhost Only")
print("-----------------------------------------------------")

# RHINO.INSIDE OR RHINO3DM
if _RHINOINSIDE:
    print("[INFO] Loading Rhino.Inside.CPython ...")
    import rhinoinside
    rhinoinside.load()
    import Rhino # NOQA402
else:
    import rhino3dm # NOQA402

# SYSTEM IF NECESSARY
if _USING_SYSTEM:
    print("[INFO] Loading System (.NET) ...")
    import System # NOQA402

# GRASSHOPPER IF NECESSARY
if _USING_GH:
    print("[INFO] Loading Grasshopper ...")
    clr.AddReference("Grasshopper.dll")
    import Grasshopper as gh # NOQA402

# KANGAROO 2 IF NECESSARY
if _USING_K2:
    print("[INFO] Loading Kangaroo2 ...")
    clr.AddReference("KangarooSolver.dll")
    import KangarooSolver as ks # NOQA402


# MODULE IMPORTS --------------------------------------------------------------

import igl # NOQA402
import numpy as np # NOQA402
import open3d as o3d # NOQA402
import potpourri3d as pp3d # NOQA402
import scipy # NOQA402
from sklearn.manifold import TSNE # NOQA402
from sklearn.decomposition import PCA # NOQA402

# LOCAL MODULE IMPORTS --------------------------------------------------------

import malt # NOQA402
from malt import hopsutilities as hsutil # NOQA402
from malt import icp # NOQA402
from malt import imgprocessing # NOQA402
from malt import intri # NOQA402
from malt import miphopper # NOQA402
from malt import shapesph # NOQA402
from malt import sshd # NOQA402


# REGSISTER FLASK AND/OR RHINOINSIDE HOPS APP ---------------------------------

if _FLASK:
    from flask import Flask # NOQA402
    flaskapp = Flask(__name__)
    hops = ExtendedHops(app=flaskapp, force_rhinoinside=_RHINOINSIDE)
elif not _FLASK and _RHINOINSIDE:
    hops = ExtendedHops(app=rhinoinside)
else:
    hops = ExtendedHops()


# HOPS COMPONENTS -------------------------------------------------------------

# GET ALL AVAILABLE COMPONENTS ////////////////////////////////////////////////

@hops.component(
    "/hops.AvailableComponents",
    name="AvailableComponents",
    nickname="Components",
    description="List all routes (URI's) of the available components",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[],
    outputs=[
        hs.HopsString("Components", "C", "All available Hops Components on this server.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsString("Description", "D", "The descriptions of the components", hs.HopsParamAccess.LIST), # NOQA501
    ])
def hops_AvailableComponentsComponent():
    comps = []
    descr = []
    for c in hops._components:
        uri = str(c)
        if not uri.startswith("/test."):
            comps.append(uri)
            descr.append(hops._components[c].description)
    return comps, descr


# GUROBI INTERFACE COMPONENTS /////////////////////////////////////////////////

@hops.component(
    "/gurobi.SolveAssignment2DPoints",
    name="SolveAssignment2DPoints",
    nickname="SolveAssignment2DPoints",
    description="Solve a 2d assignment problem given the datapoints using Gurobi.", # NOQA502
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Design", "D", "The datapoints that define the design as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Inventory", "I", "The datapoints that define the inventory from which to choose the assignment as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Cost", "C", "The cost values for the optimal solution.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveAssignment2DPointsComponent(design,
                                            inventory):

    # loop over trees and extract data points as numpy arrays
    design_p, np_design = hsutil.hops_tree_to_np_array(design)
    inventory_p, np_inventory = hsutil.hops_tree_to_np_array(inventory)

    # verify feasibility of input datapoints
    if np_design.shape[0] > np_inventory.shape[0]:
        raise ValueError("Number of Design datapoints needs to be smaller " +
                         "than or equal to number of Inventory datapoints!")

    # compute cost matrix
    cost = np.zeros((np_design.shape[0], np_inventory.shape[0]))
    for i, pt1 in enumerate(np_design):
        for j, pt2 in enumerate(np_inventory):
            cost[i, j] = np.linalg.norm(pt2 - pt1, ord=2)

    # solve the assignment problem using the gurobi interface
    assignment, assignment_cost = miphopper.solve_assignment_2d(cost)

    # return data as hops tree
    return (hsutil.np_int_array_to_hops_tree(assignment, design_p),
            hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))


@hops.component(
    "/gurobi.SolveAssignment3DPoints",
    name="SolveAssignment3DPoints",
    nickname="SolveAssignment3DPoints",
    description="Solve a 3d assignment problem given the datapoints using Gurobi.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Design", "D", "The datapoints that define the design as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Inventory", "I", "The datapoints that define the inventory from which to choose the assignment as DataTree of Numbers, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsBoolean("SimplifyCase", "S", "Simplify the 3d problem case (or at least try to) by pre-computing the minimum cost and solving the resulting 2d cost matrix.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsNumber("Cost", "C", "The cost values for the optimal solution.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveAssignment3DPointsComponent(design,
                                            inventory,
                                            simplify=False):

    # verify tree integrity
    if (not hsutil.hops_tree_verify(design) or
            not hsutil.hops_tree_verify(inventory)):
        raise ValueError("DataTree structure is inconsistent! All paths have "
                         "to be of the same shape!")

    # loop over design tree and extract data points as numpy arrays
    design_p, np_design = hsutil.hops_tree_to_np_array(design)

    # build inventory numpy array
    inventory_p, np_inventory = hsutil.hops_tree_to_np_array(inventory, True)
    inventory_shape = (len(set(p[0] for p in inventory_p)),
                       len(set(p[1] for p in inventory_p)),
                       len(np_inventory[0]))
    np_inventory_2d = np.zeros(inventory_shape)
    for path, data in zip(inventory_p, np_inventory):
        i = path[0]
        j = path[1]
        for k, d in enumerate(data):
            np_inventory_2d[i, j, k] = d

    # verify tree integrity
    if np_design.shape[0] > np_inventory_2d.shape[0]:
        raise ValueError("Number of Design datapoints needs to be smaller "
                         "than or equal to number of Inventory datapoints!")

    # simplifies the problem to a 2d assignment problem by pre-computing the
    # minimum cost and then solving a 2d assignment problem
    if simplify:
        # create empty 2d cost matrix
        cost = np.zeros((np_design.shape[0], inventory_shape[0]))
        mapping = np.zeros((np_design.shape[0], inventory_shape[0]),
                           dtype=int)

        # loop over all design objects
        for i, d_obj in enumerate(np_design):
            # loop over all objects in the inventory per design object
            for j in range(np_inventory_2d.shape[0]):
                # find minimum orientation and index of it
                pt1 = d_obj
                allcosts = [np.linalg.norm(np_inventory_2d[j, k] - pt1, ord=2)
                            for k in range(np_inventory_2d.shape[1])]
                mincost = min(allcosts)
                minidx = allcosts.index(mincost)
                # build cost matrix and store index in a mapping
                cost[i, j] = mincost
                mapping[i, j] = minidx

        # solve the assignment problem using the gurobi interface
        assignment, assignment_cost = miphopper.solve_assignment_2d(cost)

        assignment_3d = []
        for i, v in enumerate(assignment):
            assignment_3d.append((v, mapping[i, v]))
        assignment_3d = np.array(assignment_3d)

        # return data as hops tree
        return (hsutil.np_int_array_to_hops_tree(assignment_3d, design_p),
                hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))
    else:
        # create empty 3d cost martix as np array
        cost = np.zeros((np_design.shape[0],
                         inventory_shape[0],
                         inventory_shape[1]))

        # loop over all design objects
        for i, d_obj in enumerate(np_design):
            # loop over all objects in the inventory per design object
            for j in range(np_inventory_2d.shape[0]):
                # loop over orientations for every object in the inventory
                for k in range(np_inventory_2d.shape[1]):
                    pt1 = d_obj
                    pt2 = np_inventory_2d[j, k]
                    cost[i, j, k] = np.linalg.norm(pt2 - pt1, ord=2)

        # solve the assignment problem using the gurobi interface
        assignment, assignment_cost = miphopper.solve_assignment_3d(cost)

    # return data as hops tree
    return (hsutil.np_int_array_to_hops_tree(assignment, design_p),
            hsutil.np_float_array_to_hops_tree(assignment_cost, design_p))


@hops.component(
    "/gurobi.SolveCuttingStockProblem",
    name="SolveCuttingStockProblem",
    nickname="SolveCSP",
    description="Solve a cutting stock problem.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("StockLength", "SL", "Stock Length", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("StockCrossSectionLong", "SCL", "Stock Cross Section Long Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("StockCrossSectionShort", "SCS", "Stock Cross Section Short Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandLength", "DL", "Demand Length", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandCrossSectionLong", "DCL", "Demand Cross Section Long Side", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("DemandCrossSectionShort", "DCS", "Demand Cross Section Short Side", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Assignment", "A", "An optimal solution for the given assignment problem.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("NewComponents", "N", "Components produced new.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def gurobi_SolveCSPComponent(stock_len,
                             stock_cs_x,
                             stock_cs_y,
                             demand_len,
                             demand_cs_x,
                             demand_cs_y):

    # SANITIZE INPUT DATA -----------------------------------------------------

    if not len(stock_len) == len(stock_cs_x) == len(stock_cs_y):
        raise ValueError("Stock Length and Cross Section Size lists must "
                         "correspond in length!")
    if not len(demand_len) == len(demand_cs_x) == len(demand_cs_y):
        raise ValueError("Demand Length and Cross Section Size lists must "
                         "correspond in length!")

    # BUILD NP ARRAYS ---------------------------------------------------------

    m = np.column_stack((np.array([round(x, 6) for x in demand_len]),
                         np.array([round(x, 6) for x in demand_cs_x]),
                         np.array([round(x, 6) for x in demand_cs_y])))

    R = np.column_stack((np.array([round(x, 6) for x in stock_len]),
                         np.array([round(x, 6) for x in stock_cs_x]),
                         np.array([round(x, 6) for x in stock_cs_y])))

    # COMPOSE N ON BASIS OF M -------------------------------------------------

    cs_set = sorted(list(set([(x[1], x[2]) for x in m])), reverse=True)
    N = np.array([(float("inf"), x[0], x[1]) for x in cs_set])

    # RUN CUTTING STOCK OPTIMIZATION ------------------------------------------

    optimisation_result = miphopper.solve_csp(m, R, N)

    # RETURN THE OPTIMIZATION RESULTS -----------------------------------------

    return ([float(int(x[1])) for x in optimisation_result],
            hsutil.np_float_array_to_hops_tree(N))


# ITERATIVE CLOSEST POINT /////////////////////////////////////////////////////

@hops.component(
    "/icp.RegisterPointClouds",
    name="RegisterPointClouds",
    nickname="Register",
    description="Register a Scene PointCloud with a given Model PointCloud.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("ScenePoints", "S", "ScenePoints to evaluate", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsPoint("ModelPoints", "M", "ModelPoints for ICP", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Threshold", "T", "Threshold for convergence", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("MaxIterations", "I", "Maximum iterations", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("NNAlgorithm", "A", "Algorithm used for nearest neighbor computation, can be 0 (KNN) or 1 (Hungarian). Defaults to KNN.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsPoint("RegisteredPoints", "R", "Registered ScenePoints", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Transform", "X", "Transformation Matrix", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Error", "E", "Mean Error of ICP operation", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Iterations", "I", "Iterations before termination.", hs.HopsParamAccess.ITEM), # NOQA501
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
    iters = float(res[2])

    # return the results
    return (transformed_pts, xformlist, err, iters)


# LIBIGL //////////////////////////////////////////////////////////////////////

@hops.component(
    "/igl.MeshIsocurves",
    name="MeshIsocurves",
    nickname="MeshIsoC",
    description="Compute isocurves based on a function using libigl",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute isocurves on.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Values", "V", "The function to compute as a list of values at each vertex position of the mesh.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Count", "C", "Number of Isocurves", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsLine("Isolines", "I", "The resulting isolines.", hs.HopsParamAccess.LIST), # NOQA501
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

    isoLines = []
    for edge in isoE:
        isoLines.append(Rhino.Geometry.Line(isoV[int(edge[0])],
                                            isoV[int(edge[1])]))

    return isoLines


# INTRINSIC TRIANGULATIONS ////////////////////////////////////////////////////

@hops.component(
    "/intri.IntrinsicTriangulation",
    name="IntrinsicTriangulation",
    nickname="InTri",
    description="Compute intrinsic triangulation of a triangle mesh.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to create intrinsic triangulation for.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("IntrinsicTriangulation", "T", "The resulting Mesh with an intrinsic triangulation.", hs.HopsParamAccess.ITEM), # NOQA501
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
    [intri_mesh.Faces.AddFace(int(f[0]), int(f[1]), int(f[2])) for f in F]

    # compute normals and compact
    intri_mesh.UnifyNormals()
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
    icon="resources/icons/220204_malt_icon.png",
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
    "/open3d.AlphaShape",
    name="AlphaShape",
    nickname="AlphaShape",
    description="Construct a Mesh from a PointCloud using Open3D alpha shape algorithm.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Alpha", "A", "The Alpha value for the algorithm. Defaults to 1.0.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting triangle Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def open3d_AlphaShapeComponent(points, alpha=1.0):

    # convert point list to np array
    np_points = np.array([[pt.X, pt.Y, pt.Z] for pt in points])

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    # estimate the normals
    pointcloud.estimate_normals()

    # compute convex hull triangle mesh
    a_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                                                                    pointcloud,
                                                                    alpha)

    # create rhino mesh from o3d output and add vertices and faces
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                            np.asarray(a_shape.vertices),
                                            np.asarray(a_shape.triangles),
                                            Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.ConvexHull",
    name="ConvexHull",
    nickname="ConvexHull",
    description="Construct a Mesh from a PointCloud using Open3D convex hull.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting triangle Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def open3d_ConvexHullComponent(points):

    # convert point list to np array
    np_points = np.array([[pt.X, pt.Y, pt.Z] for pt in points])

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    # estimate the normals
    pointcloud.estimate_normals()

    # compute convex hull triangle mesh
    convex_hull = pointcloud.compute_convex_hull()

    # create rhino mesh from o3d output and add vertices and faces
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                        np.asarray(convex_hull[0].vertices),
                                        np.asarray(convex_hull[0].triangles),
                                        Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.BallPivotingMesh",
    name="BallPivotingMesh",
    nickname="BallPivotingMesh",
    description="Construct a Mesh from a PointCloud and corresponding normals using Open3D ball pivoting reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Depth", "D", "Depth parameter for the poisson algorithm. Defaults to 8.", hs.HopsParamAccess.ITEM), # NOQA501
        ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def open3d_BallPivotingMeshComponent(points,
                                     triangles=100000):

    # convert point list to np array
    np_points = hsutil.rhino_points_to_np_array(points)

    # create pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np_points)

    # estimate normals of the pcd
    pointcloud.estimate_normals()

    # compute radius for bpa algorithm
    distances = pointcloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    # create poisson mesh reconstruction
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pointcloud,
                        o3d.utility.DoubleVector([radius,
                                                  radius * 2,
                                                  radius * 3]))

    # downsample mesh to a lower number of triangles
    if triangles > 0:
        bpa_mesh = bpa_mesh.simplify_quadric_decimation(triangles)

    # remove artifacts and ensure mesh consistency
    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    # create rhino mesh from o3d output and vertices and faces
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                        np.asarray(bpa_mesh.vertices),
                                        np.asarray(bpa_mesh.triangles),
                                        Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.BallPivotingMeshNormals",
    name="BallPivotingMeshNormals",
    nickname="BallPivotingMeshNormals",
    description="Construct a Mesh from a PointCloud and corresponding normals using Open3D ball pivoting reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("Points", "P", "The PointClouds Points", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsVector("Normals", "N", "The Normals of the pointcloud to be used in reconstruction", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsInteger("Triangles", "T", "Number of triangles the resulting mesh gets downsampled to. No downsampling will happen if 0 is supplied.. Defaults to 100000.", hs.HopsParamAccess.ITEM), # NOQA501
        ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The resulting Mesh.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def open3d_BallPivotingMeshNormalsComponent(points,
                                            normals,
                                            triangles=100000):

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

    # compute radius for bpa algorithm
    distances = pointcloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    # create poisson mesh reconstruction
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pointcloud,
                        o3d.utility.DoubleVector([radius,
                                                  radius * 2,
                                                  radius * 3]))

    # downsample mesh to a lower number of triangles
    if triangles > 0:
        bpa_mesh = bpa_mesh.simplify_quadric_decimation(triangles)

    # remove artifacts and ensure mesh consistency
    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    # create rhino mesh from results
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                            np.asarray(bpa_mesh.vertices),
                                            np.asarray(bpa_mesh.triangles),
                                            Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.PoissonMesh",
    name="PoissonMesh",
    nickname="PoissonMesh",
    description="Construct a Mesh from a PointCloud using Open3D poisson mesh reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
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
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                            np.asarray(p_mesh_crop.vertices),
                                            np.asarray(p_mesh_crop.triangles),
                                            Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


@hops.component(
    "/open3d.PoissonMeshNormals",
    name="PoissonMeshNormals",
    nickname="PoissonMeshNormals",
    description="Construct a Mesh from a PointCloud and corresponding normals using Open3D poisson mesh reconstruction.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
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
    rhino_mesh = hsutil.np_arrays_to_rhino_triangle_mesh(
                                            np.asarray(p_mesh_crop.vertices),
                                            np.asarray(p_mesh_crop.triangles),
                                            Rhino=Rhino)

    # return the rhino mesh
    return rhino_mesh


# OPENCV //////////////////////////////////////////////////////////////////////

@hops.component(
    "/opencv.CalibrateCameraCapture",
    name="CalibrateCameraCapture",
    nickname="CalCamCap",
    description="Calibrate a connected Webcam for contour detection using OpenCV.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Run", "R", "Run the capturing and contour detection routine.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Device", "D", "The identifier of the capture device to use. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "The width of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Height", "H", "The height of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Transform", "T", "The transformation matrix.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def opencv_CalibrateCameraCaptureComponent(run,
                                           device=0,
                                           width=3780,
                                           height=1890):

    # initilize identity matrix
    xform = np.eye(3)

    # run camera calibration
    if run:
        image = imgprocessing.capture_image(device)
        xform = imgprocessing.calibrate_camera_image(image, width, height)
    # get hops-compatible tree structure
    rhinoxform = hsutil.np_float_array_to_hops_tree(xform)

    return rhinoxform


@hops.component(
    "/opencv.CalibrateCameraFile",
    name="CalibrateCameraFile",
    nickname="CalCamFile",
    description="Calibrate a camera based on an image for contour detection using OpenCV.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Run", "R", "Run the calibration routine.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsString("FilePath", "F", "The filepath of the image.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "The width of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Height", "H", "The height of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Transform", "T", "The transformation matrix.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def opencv_CalibrateCameraFileComponent(run,
                                        filepath="",
                                        width=3780,
                                        height=1890):
    # initilize identity matrix
    xform = np.eye(3)

    # run camera calibration
    if run:
        xform = imgprocessing.calibrate_camera_file(filepath, width, height)

    # get hops-compatible tree structure
    rhinoxform = hsutil.np_float_array_to_hops_tree(xform)

    return rhinoxform


@hops.component(
    "/opencv.LoadCameraXForm",
    name="LoadCameraXForm",
    nickname="LoadCamXF",
    description="Load a camera perspective transformation from a file.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsString("Filepath", "F", "The filepath of the transformation.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsString("Log", "L", "The logging output of this component.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Transform", "T", "The transformation matrix.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def opencv_LoadCameraXForm(filepath: str = ""):

    log = []
    rhinoxform = {}

    if not filepath or not os.path.isfile(filepath):
        filepath = hsutil.sanitize_path(os.path.join(malt.IMGDIR, "xform.yml"))
        log.append("No filepath supplied, will load default file...")

    try:
        log.append("Loading transformation matrix from file:")
        log.append(filepath)
        xform = imgprocessing.load_perspective_xform(filepath)
        rhinoxform = hsutil.np_float_array_to_hops_tree(xform)
    except SystemError:
        raise ValueError(("Could not read perspective transformation "
                          "from {0}. Make sure the file is obtained by "
                          "calling compute_perspective_xform()!"))

    return log, rhinoxform


@hops.component(
    "/opencv.DetectContoursCapture",
    name="DetectContoursCapture",
    nickname="DetConCap",
    description="Capture an image from a connected camera and detect contours using OpenCV.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Run", "R", "Run the capturing and contour detection routine.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("BinaryThreshold", "B", "The threshold for binary (black & white) conversion of the image. Defaults to 127.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("AreaThreshold", "T", "The area threshold for filtering the returned contours in pixels. Deactivated if set to 0. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Device", "D", "The identifier of the capture device to use. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "The width of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Height", "H", "The height of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("ChainApproximation", "C", "The chain approximation to use during contour detection. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("Invert", "I", "If True, threshold image will be inverted. Use the invert function to detect black objects on white background.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("ExtOnly", "E", "If True, only external contours will be returned.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Transform", "X", "The transformation parameters from camera calibration.", hs.HopsParamAccess.TREE), # NOQA501
    ],
    outputs=[
        hs.HopsCurve("Boundary", "B", "The detected image boundary.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsCurve("Contours", "C", "The detected contours as Polylines.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def opencv_DetectContoursCaptureComponent(run,
                                          bthresh=127,
                                          athresh=0.0,
                                          device=0,
                                          width=3780,
                                          height=1890,
                                          chain=0,
                                          invert=False,
                                          external=True,
                                          xformtree=None):

    if run:
        # capture an image for contour detection
        image = imgprocessing.capture_image(device)

        # retrieve transformation matrix from calibrate camera tree input
        xform = hsutil.hops_tree_to_np_array(xformtree)[1]

        # create warped image
        warped_img = imgprocessing.warp_image(image, xform, width, height)

        # run contour detection using opencv
        warped_img, contours = imgprocessing.detect_contours_from_image(
                                        warped_img,
                                        bthresh,
                                        athresh,
                                        chain,
                                        invert,
                                        external)

        # compute scaling factor for results
        h_dst, w_dst, c_dst = warped_img.shape
        scalingfactor = width / w_dst

        # construct polylines from contour output
        plcs = []
        for cnt in contours:
            # create .NET list because Polyline constructor won't correctly
            # handle python lists (it took a while to find that out....)
            if len(cnt) >= 2:
                ptL = System.Collections.Generic.List[Rhino.Geometry.Point3d]()
                # add contour points to .NET list
                [ptL.Add(Rhino.Geometry.Point3d(float(pt[0][0]),
                                                float(pt[0][1]),
                                                0.0)) for pt in cnt]
                # add first point again to close polyline
                ptL.Add(Rhino.Geometry.Point3d(float(cnt[0][0][0]),
                                               float(cnt[0][0][1]),
                                               0.0))
                # create polylinecurve from .NET list and scale with factor
                plc = Rhino.Geometry.PolylineCurve(ptL)
                plc.Scale(scalingfactor)
                # append to output list
                plcs.append(plc)

        # create boundary of image as polyline curve for reference
        bPts = System.Collections.Generic.List[Rhino.Geometry.Point3d]()
        # add boundary points to .NET list
        bPts.Add(Rhino.Geometry.Point3d(0.0, 0.0, 0.0))
        bPts.Add(Rhino.Geometry.Point3d(float(w_dst), 0.0, 0.0))
        bPts.Add(Rhino.Geometry.Point3d(float(w_dst), float(h_dst), 0.0))
        bPts.Add(Rhino.Geometry.Point3d(0.0, float(h_dst), 0.0))
        bPts.Add(Rhino.Geometry.Point3d(0.0, 0.0, 0.0))
        # create polylinecurve from .NET list and scale with factor
        boundary = Rhino.Geometry.PolylineCurve(bPts)
        boundary.Scale(scalingfactor)

        # return output
        return ([boundary], plcs)

    return ([], [])


@hops.component(
    "/opencv.DetectContoursFile",
    name="DetectContoursFile",
    nickname="DetConFile",
    description="Use an image file and detect contours using OpenCV.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsBoolean("Run", "R", "Run the capturing and contour detection routine.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsString("FilePath", "F", "The filepath of the image.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("BinaryThreshold", "B", "The threshold for binary (black & white) conversion of the image. Defaults to 127.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("AreaThreshold", "T", "The area threshold for filtering the returned contours in pixels. Deactivated if set to 0. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Width", "W", "The width of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Height", "H", "The height of the working area.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("ChainApproximation", "C", "The chain approximation to use during contour detection. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("Invert", "I", "If True, threshold image will be inverted. Use the invert function to detect black objects on white background.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsBoolean("ExtOnly", "E", "If True, only external contours will be returned.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Transform", "X", "The transformation parameters from camera calibration.", hs.HopsParamAccess.TREE), # NOQA501
    ],
    outputs=[
        hs.HopsCurve("Boundary", "B", "The detected image boundary.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsCurve("Contours", "C", "The detected contours as Polylines.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def opencv_DetectContoursFileComponent(run,
                                       filepath="",
                                       bthresh=127,
                                       athresh=0.0,
                                       width=1000,
                                       height=1000,
                                       chain=0,
                                       invert=False,
                                       external=True,
                                       xformtree=None):

    if run:
        # read image from filepath
        image = imgprocessing.read_image(filepath)

        # retrieve transformation matrix from calibrate camera tree input
        xform = hsutil.hops_tree_to_np_array(xformtree)[1]

        # create warped image
        warped_img = imgprocessing.warp_image(image, xform, width, height)

        # run contour detection using opencv
        warped_img, contours = imgprocessing.detect_contours_from_image(
                                        warped_img,
                                        bthresh,
                                        athresh,
                                        chain,
                                        invert,
                                        external)

        # compute scaling factor for results
        h_dst, w_dst, c_dst = warped_img.shape
        scalingfactor = width / w_dst

        # construct polylines from contour output
        plcs = []
        for cnt in contours:
            # create .NET list because Polyline constructor won't correctly
            # handle python lists (it took a while to find that out....)
            if len(cnt) >= 2:
                ptL = System.Collections.Generic.List[Rhino.Geometry.Point3d]()
                # add contour points to .NET list
                [ptL.Add(Rhino.Geometry.Point3d(float(pt[0][0]),
                                                float(pt[0][1]),
                                                0.0)) for pt in cnt]
                # add first point again to close polyline
                ptL.Add(Rhino.Geometry.Point3d(float(cnt[0][0][0]),
                                               float(cnt[0][0][1]),
                                               0.0))
                # create polylinecurve from .NET list and scale with factor
                plc = Rhino.Geometry.PolylineCurve(ptL)
                plc.Scale(scalingfactor)
                # append to output list
                plcs.append(plc)

        # create boundary of image as polyline curve for reference
        bPts = System.Collections.Generic.List[Rhino.Geometry.Point3d]()
        # add boundary points to .NET list
        bPts.Add(Rhino.Geometry.Point3d(0.0, 0.0, 0.0))
        bPts.Add(Rhino.Geometry.Point3d(float(w_dst), 0.0, 0.0))
        bPts.Add(Rhino.Geometry.Point3d(float(w_dst), float(h_dst), 0.0))
        bPts.Add(Rhino.Geometry.Point3d(0.0, float(h_dst), 0.0))
        bPts.Add(Rhino.Geometry.Point3d(0.0, 0.0, 0.0))
        # create polylinecurve from .NET list and scale with factor
        boundary = Rhino.Geometry.PolylineCurve(bPts)
        boundary.Scale(scalingfactor)

        # return output
        return ([boundary], plcs)

    return ([], [])


# POTPOURRI3D /////////////////////////////////////////////////////////////////

@hops.component(
    "/pp3d.MeshHeatMethodDistance",
    name="MeshHeatMethodDistance",
    nickname="MeshHeatDist",
    description="Compute geodesic distances using the heat method.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute geodesic distances on.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Sources", "S", "The indices of the source vertices from which to compute geodesic distances using the heat method. Defaults to 0.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("TextureScale", "T", "The texturescale used for setting the normalized values to the texture coordinates.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The mesh with texture coordinates set to the normalized values.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("Distances", "D", "The geodesic distances to all mesh vertices.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Values", "V", "The normalized values for every mesh vertex.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def pp3d_HeatMethodDistanceComponent(mesh,
                                     sources,
                                     texture_scale=1.0):

    # ensure that the user picked a feasible source
    for v in list(sources):
        assert v <= mesh.Vertices.Count - 1, ("The index of the "
                                              "source vertex cannot "
                                              "exceed the vertex "
                                              "count!")

    # get vertex and face array
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # compute geodesic heat distances
    if len(sources) == 1:
        geodists = [float(x) for x in
                    pp3d.compute_distance(V, F, sources[0])]
    else:
        geodists = [float(x) for x in
                    pp3d.compute_distance_multisource(V, F, sources)]

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


@hops.component(
    "/pp3d.MeshVectorHeatExtendScalar",
    name="MeshVectorHeatExtendScalar",
    nickname="MeshVecHeatExtScal",
    description="Extend scalar values along a mesh using the vector heat method.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to extend scalar on.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Sources", "S", "The indices of the source vertices from which to extend the scalar values.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("Values", "V", "The scalar values to extend per source vertex.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("TextureScale", "T", "The texturescale used for setting the normalized values to the texture coordinates.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsMesh("Mesh", "M", "The mesh with texture coordinates set to the normalized values.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("ExtendedScalars", "E", "The extended scalar values for every mesh vertex.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsNumber("NormalizedValues", "N", "The normalized extended scalar values for every mesh vertex.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def pp3d_MeshVectorHeatExtendScalarComponent(mesh,
                                             sources,
                                             values,
                                             texture_scale=1.0):

    # ensure that the user picked feasible sources
    for v in list(sources):
        assert v >= 0, "Vertex index cannot be negative!"
        assert v <= mesh.Vertices.Count - 1, ("The index of the "
                                              "source vertex cannot "
                                              "exceed the vertex "
                                              "count!")

    # sanitize input list lengths
    assert len(sources) == len(values), ("Number of sources and tangent "
                                         "vectors has to correspond!")

    # get vertex and face array
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # Init vector heat solver
    vhmsolver = pp3d.MeshVectorHeatSolver(V, F)

    # extend the scalar
    extended_scalars = [float(v) for v in
                        vhmsolver.extend_scalar(sources, values)]

    # normalize all values for setting texture coordinates
    vmin = min(extended_scalars)
    vmax = max(extended_scalars)
    mult = 1.0 / (vmax - vmin)
    nrmvalues = [mult * (v - vmin) for v in extended_scalars]

    # set texture coordinates of output mesh
    out_mesh = mesh.Duplicate()
    for i in range(out_mesh.Vertices.Count):
        out_mesh.TextureCoordinates.SetTextureCoordinate(
                    i,
                    Rhino.Geometry.Point2f(0.0, nrmvalues[i] * texture_scale))

    return out_mesh, extended_scalars, nrmvalues


@hops.component(
    "/pp3d.MeshVectorHeatParallelTransport",
    name="MeshVectorHeatParallelTransport",
    nickname="MeshVecHeatTransp",
    description="Parallel transport a vector along the surface.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to use for parallel transport.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Sources", "S", "The indices of the source vertices from which to transport the tangent vectors.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsVector("Vectors", "V", "The 2d tangent vectors to parallel transport per source vertex.", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[
        hs.HopsVector("TangentFrames", "F", "The tangent frames used by the solver.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsVector("TransportVectors", "T", "The tangent vectors after parallel transport.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsVector("MappedVectors", "M", "The tangent vectors after parallel transport mapped to 3D.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def pp3d_MeshVectorHeatParallelTransportComponent(mesh,
                                                  sources,
                                                  vectors):

    # ensure that the user picked feasible sources
    for v in list(sources):
        assert v >= 0, "Vertex index cannot be negative!"
        assert v <= mesh.Vertices.Count - 1, ("The index of the "
                                              "source vertex cannot "
                                              "exceed the vertex "
                                              "count!")

    # sanitize input list lengths
    assert len(sources) == len(vectors), ("Number of sources and tangent "
                                          "vectors has to correspond!")

    # get vertex and face array
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # sanitize rhino vectors to 2d tangent space
    tvectors = []
    for v in vectors:
        tvectors.append([v.X, v.Y])

    # init vector heat solver
    vhmsolver = pp3d.MeshVectorHeatSolver(V, F)

    # get tangent frames and convert them to rhino planes
    basisX, basisY, basisN = vhmsolver.get_tangent_frames()
    frames = {}
    for i, (bX, bY, bN) in enumerate(zip(basisX, basisY, basisN)):
        origin = Rhino.Geometry.Vector3d(
                        Rhino.Geometry.Point3d(mesh.Vertices[i]))
        xaxis = Rhino.Geometry.Vector3d(bX[0], bX[1], bX[2])
        yaxis = Rhino.Geometry.Vector3d(bY[0], bY[1], bY[2])
        frames["{%s}" % str(i)] = (origin, xaxis, yaxis)

    # extend the vector via parallel transport
    if len(sources) > 1:
        ext_vectors = vhmsolver.transport_tangent_vectors(sources,
                                                          tvectors)
    else:
        ext_vectors = vhmsolver.transport_tangent_vector(sources[0],
                                                         tvectors[0])

    # map extended vectors to 3d space
    ext_vectors_3d = (ext_vectors[:, 0, np.newaxis] * basisX +
                      ext_vectors[:, 1, np.newaxis] * basisY)

    # convert vectors to rhino
    rh_vectors = hsutil.np_array_to_rhino_vectors(ext_vectors, Rhino)
    rh_vectors_3d = hsutil.np_array_to_rhino_vectors(ext_vectors_3d, Rhino)

    # return the results of the parallel transport
    return frames, rh_vectors, rh_vectors_3d


# SKLEARN /////////////////////////////////////////////////////////////////////

@hops.component(
    "/sklearn.TSNE",
    name="TSNE",
    nickname="TSNE",
    description="T-distributed Stochastic Neighbor Embedding.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Data", "D", "Point Data to be reduced using t-SNE as a DataTree, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsInteger("Components", "N", "Dimension of the embedded space.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Perplexity", "P", "The perplexity is related to the number of nearest neighbors that are used in other manifold learning algorithms. Consider selecting a value between 5 and 50. Defaults to 30.", hs.HopsParamAccess.ITEM, ), # NOQA501
        hs.HopsNumber("EarlyExaggeration", "E", "Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. Defaults to 12.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("LearningRate", "R", "The learning rate for t-SNE is usually in the range (10.0, 1000.0). Defaults to 200.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Iterations", "I", "Maximum number of iterations for the optimization. Should be at least 250. Defaults to 1000.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Method", "M", "Barnes-Hut approximation (0) runs in O(NlogN) time. Exact method (1) will run on the slower, but exact, algorithm in O(N^2) time. Defaults to 0.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Initialization", "I", "Initialization method. Random (0) or PCA (1).", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("RandomSeed", "S", "Determines the random number generator. Pass an int for reproducible results across multiple function calls. Note that different initializations might result in different local minima of the cost function. Defaults to None.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Points", "T", "The transformed points", hs.HopsParamAccess.TREE), # NOQA501
    ])
def sklearn_TSNEComponent(data,
                          n_components=2,
                          perplexity=30,
                          early_exaggeration=12.0,
                          learning_rate=200.0,
                          n_iter=1000,
                          method=0,
                          init=0,
                          rnd_seed=0):
    # loop over tree and extract data points
    paths, np_data = hsutil.hops_tree_to_np_array(data)
    # convert method string
    if method <= 0:
        method_str = "barnes_hut"
    else:
        method_str = "exact"
    if init <= 0:
        init_str = "random"
    else:
        init_str = "pca"
    # initialize t-SNE solver class
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=rnd_seed,
                method=method_str,
                init=init_str)
    # run t-SNE solver on incoming data
    tsne_result = tsne.fit_transform(np_data)
    # return data as hops tree (dict)
    return hsutil.np_float_array_to_hops_tree(tsne_result, paths)


@hops.component(
    "/sklearn.PCA",
    name="PCA",
    nickname="PCA",
    description="Principal component analysis.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("Data", "D", "Point Data to be reduced using PCA as a DataTree, where each Branch represents one Point.", hs.HopsParamAccess.TREE), # NOQA501
        hs.HopsInteger("Components", "N", "Number of components (dimensions) to keep.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("Points", "T", "The transformed points", hs.HopsParamAccess.TREE), # NOQA501
    ])
def sklearn_PCAComponent(data,
                         n_components=2):
    # loop over tree and extract data points
    paths, np_data = hsutil.hops_tree_to_np_array(data)
    # initialize PCA solver class
    pca = PCA(n_components=n_components)
    # run PCA solver on incoming data
    pca_result = pca.fit_transform(np_data)
    # return data as hops tree (dict)
    return hsutil.np_float_array_to_hops_tree(pca_result, paths)


# SPHERICAL HARMONICS SHAPE DESCRIPTOR ////////////////////////////////////////

@hops.component(
    "/sshd.MeshSphericalHarmonicsDescriptor",
    name="MeshSphericaHarmonicsDescriptor",
    nickname="MeshSHD",
    description="Description of mesh using a complex function on the sphere.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute the shape descriptor for.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsInteger("Dimensions", "D", "Number of dimensions/coefficients for the computation. Defaults to 13.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("FeatureVector", "F", "The feature vector for the shape descriptor.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def sshd_MeshSphericalHarmonicsDescriptorComponent(mesh, dims=13):
    # check if mesh is all triangles
    if mesh.Faces.QuadCount > 0:
        raise ValueError("Mesh has to be triangular!")

    # get np arrays of vertices and faces
    V, F = hsutil.rhino_mesh_to_np_arrays(mesh)

    # compute shape descriptor
    sdescr = sshd.descriptorCS(V, F, coef_num_sqrt=dims)

    # convert descriptor values to floats
    sdescr = [float(x) for x in sdescr]

    # return results
    return sdescr


@hops.component(
    "/shapesph.MeshSphericalHarmonicsDescriptorRI",
    name="MeshSphericaHarmonicsDescriptorRI",
    nickname="MeshSHD",
    description="Description of mesh using rotation invariant spherical harmonics descriptor", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsMesh("Mesh", "M", "The triangle mesh to compute the shape descriptor for.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("FeatureVector", "F", "The feature vector for the shape descriptor.", hs.HopsParamAccess.LIST), # NOQA501
    ])
def shapesph_MeshSphericalHarmonicsDescriptorRIComponent(mesh):
    # check if mesh is all triangles
    if mesh.Faces.QuadCount > 0:
        raise ValueError("Mesh has to be triangular!")

    # get plyfile elements of vertices and faces
    V, F = hsutil.rhino_mesh_to_ply_elements(mesh)

    # compute shape descriptor
    sdescr = shapesph.compute_descriptor(V, F)

    # convert descriptor values to floats
    sdescr = [float(x) for x in sdescr]

    # return results
    return sdescr


# UTILS ///////////////////////////////////////////////////////////////////////

@hops.component(
    "/utils.HausdorffDistance",
    name="HausdorffDistance",
    nickname="HDist",
    description="Hausdorff distance between two polylines.", # NOQA501
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPoint("CurvePointsA", "A", "The points of the first polyline.", hs.HopsParamAccess.LIST), # NOQA501
        hs.HopsPoint("CurvePointsB", "B", "The points of the second polyline.", hs.HopsParamAccess.LIST), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("DirectedHausdorffDistance", "D", "The directed hausdorff distance between the two polylines from A to B.", hs.HopsParamAccess.ITEM), # NOQA501
        hs.HopsNumber("SymmetricHausdorffDistance", "S", "The symmetric (maximum) hausdorff distance between the two polylines A and B.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def utils_HausdorffDistanceComponent(curveA, curveB):

    u = hsutil.rhino_points_to_np_array(curveA)
    v = hsutil.rhino_points_to_np_array(curveB)

    dA, iuA, ivA = scipy.spatial.distance.directed_hausdorff(u, v)
    dB, iuB, ivB = scipy.spatial.distance.directed_hausdorff(v, u)

    return dA, max(dA, dB)


# TEST AND VERIFICATION COMPONENTS ////////////////////////////////////////////

@hops.component(
    "/test.DataTree",
    name="tDataTree",
    nickname="tDataTree",
    description="Test DataTree input/output.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsNumber("InTree", "I", "Input Tree.", hs.HopsParamAccess.TREE), # NOQA501
    ],
    outputs=[
        hs.HopsNumber("OutTree", "O", "Output Tree.", hs.HopsParamAccess.TREE), # NOQA501
    ])
def test_DataTreeComponent(tree):

    if not tree:
        tree = {}
        tree["0;0"] = [0.0]
        tree["0;1"] = [0, 1]
        tree["0;2"] = ["a", "b", "c"]
        tree["0;3"] = [0.0, "abc", True]
    return tree


@hops.component(
    "/test.Circle",
    name="tCircle",
    nickname="tCircle",
    description="Test Circle Param.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsCircle("InCircle", "I", "Input Circle.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsCircle("OutCircle", "O", "Output Circle.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def test_CircleComponent(circle):

    if not circle:
        circle = Rhino.Geometry.Circle(Rhino.Geometry.Plane.WorldXY,
                                       1.0)
    return circle


@hops.component(
    "/test.Plane",
    name="tPlane",
    nickname="tPlane",
    description="Test Plane Param.",
    category=None,
    subcategory=None,
    icon="resources/icons/220204_malt_icon.png",
    inputs=[
        hs.HopsPlane("InPlane", "I", "Input Plane.", hs.HopsParamAccess.ITEM), # NOQA501
    ],
    outputs=[
        hs.HopsPlane("OutPlane", "O", "Output Plane.", hs.HopsParamAccess.ITEM), # NOQA501
    ])
def test_PlaneComponent(plane):

    if not plane:
        plane = Rhino.Geometry.Plane.WorldXY
    return plane


# RUN HOPS APP AS EITHER FLASK OR DEFAULT -------------------------------------

if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("[INFO] Available Hops Components on this Server:")
    [print("{0} -> {1}".format(c, hops._components[c].description))
     for c in hops._components if not str(c).startswith("/test.")]
    print("-----------------------------------------------------")
    if type(hops) == hs.HopsFlask:
        if _NETWORK_ACCESS:
            flaskapp.run(debug=_DEBUG, host="0.0.0.0")
        else:
            flaskapp.run(debug=_DEBUG)
    else:
        hops.start(debug=_DEBUG)
