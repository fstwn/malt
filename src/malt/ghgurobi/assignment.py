# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------
import os

# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import gurobipy as gp
import numpy as np

# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import hopsutilities as hsutil


# FUNCTION DEFINITIONS --------------------------------------------------------

def solve_assignment_2d(cost: np.array, verbose: bool = False):
    """
    Solves an assignment problem defined by a given 2d cost matrix using 
    Gurobi.
    
    The cost matrix does not need to be square, but num of columns
    (tasks/inventory) has to be larger or equal to num of rows
    (workers/design).
    """

    # print info and create profiler
    print("[GHGUROBI] Building Gurobi Model...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("2D Assignment")

    # add binary decision variables for num of rows and num of cols of cost
    # matrix
    # NOTE: shape[0] is the row index, shape[1] is the column index!
    select = model.addVars(cost.shape[0],
                           cost.shape[1],
                           vtype=gp.GRB.BINARY,
                           name="select")

    # add constraint so that each worker i can perform at maximum one task j
    # sum_{i = 1}^{n} x_{ij} <= 1 for all j
    model.addConstrs((gp.quicksum(select[i, j]
                      for i in range(cost.shape[0])) <= 1
                     for j in range(cost.shape[1])), name="workload")

    # add constraint so that each task j is performed by exactly one worker i
    # sum_{j = 1}^{n} x_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(select[i, j]
                      for j in range(cost.shape[1])) == 1
                     for i in range(cost.shape[0])), name="completion")

    # set objective function as sum of cost times the decision variable
    # sum_{i = 1}^{n} sum_{j = 1}^{n} c_{ij} x_{ij}
    model.setObjective(gp.quicksum(cost[i, j] * select[i, j]
                                   for i in range(cost.shape[0])
                                   for j in range(cost.shape[1])),
                       gp.GRB.MINIMIZE)

    # don't print all of the info...
    if not verbose:
        model.setParam("OutputFlag", False)

    # stop the profiler and print time elapsed for building model
    print("[GHGUROBI] Building model took {0} ms".format(timer.rawstop()))

    # optimize the model and time it with the simple profiler
    timer.start()
    model.optimize()

    # stop profiler and print time elaspsed for solving
    print("[GHGUROBI] Solving model took {0} ms".format(timer.rawstop()))

    # collect the results of the assignment
    assignment_result = []
    assignment_cost = []
    for k in select.keys():
        if select[k].x > 0:
            i = k[0]
            j = k[1]
            assignment_result.append(j)
            assignment_cost.append(cost[i, j])
            if verbose:
                print("Design {0}: Inventory {1} // "
                      "Cost: {2}".format(i, j, cost[i, j]))

    # return the optimal solution
    return np.array(assignment_result), np.array(assignment_cost)


def solve_assignment_3d(cost: np.array, verbose: bool = False):
    """
    Solves an assignment problem defined by a given 3d cost matrix using
    Gurobi.
    """

    # print info and create profiler
    print("[GHGUROBI] Building Gurobi Model...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("3D Assignment")

    # add binary decision variables
    select = model.addVars(cost.shape[0],
                           cost.shape[1],
                           vtype=gp.GRB.BINARY,
                           name="select")

    orient = model.addVars(cost.shape[1],
                           cost.shape[2],
                           vtype=gp.GRB.BINARY,
                           name="orient")

    # add constraint so that each design i can be filled by at maximum one
    # inventory object j
    # sum_{i = 1}^{n} x_{ij} <= 1 for all j
    model.addConstrs((gp.quicksum(select[i, j]
                      for i in range(cost.shape[0])) <= 1
                     for j in range(cost.shape[1])), name="select_workload")

    model.addConstrs((gp.quicksum(orient[j, k]
                      for j in range(cost.shape[1])) <= 1
                     for k in range(cost.shape[2])), name="orient_workload")

    # add constraint so that each inventory object j can be used by exactly one
    # design object i
    # sum_{j = 1}^{n} x_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(select[i, j]
                      for j in range(cost.shape[1])) == 1
                     for i in range(cost.shape[0])), name="select_completion")

    model.addConstrs((gp.quicksum(orient[j, k]
                      for k in range(cost.shape[2])) == 1
                     for j in range(cost.shape[1])), name="orient_completion")

    # set objective function as sum of cost times the decision variable
    # sum_{i = 1}^{n} sum_{j = 1}^{n} c_{ij} x_{ij}
    model.setObjective(gp.quicksum(cost[i, j, k] * select[i, j] * orient[j, k]
                                   for i in range(cost.shape[0])
                                   for j in range(cost.shape[1])
                                   for k in range(cost.shape[2])),
                       gp.GRB.MINIMIZE)

    # don't print all of the info...
    if not verbose:
        model.setParam("OutputFlag", False)

    # stop the profiler and print time elapsed for building model
    print("[GHGUROBI] Building model took {0} ms".format(timer.rawstop()))

    # optimize the model and time it with the simple profiler
    timer.start()
    model.optimize()

    # stop profiler and print time elaspsed for solving
    print("[GHGUROBI] Solving model took {0} ms".format(timer.rawstop()))

    # collect the results of the assignment
    assignment_result = []
    assignment_cost = []
    for sk in select.keys():
        if select[sk].x > 0:
            for ok in (key for key in orient.keys() if key[0] == sk[1]):
                if orient[ok].x > 0:
                    i = sk[0]
                    j = sk[1]
                    k = ok[1]
                    cijk = cost[i, j, k]
                    assignment_result.append((j, k))
                    assignment_cost.append(cijk)
                    if verbose:
                        print("Design {0}: Iventory {1} -> Orientation {2} // "
                              "Cost: {3}".format(i, j, k, cijk))

    # return the optimal solution
    return np.array(assignment_result), np.array(assignment_cost)


# TESTING ---------------------------------------------------------------------

if __name__ == "__main__":
    # TEST 2D-ASSIGNMENT USING RANDOM COST MATRIX
    cost_matrix = np.random.uniform(1, 100, (10, 100))
    solve_assignment_2d(cost_matrix, verbose=True)

    # TEST 3D-ASSIGNMENT
    _HERE = os.path.dirname(__file__)

    # DATA FORMAT OF INVENTORY SAMPLE DATA:
    # 4 OBJECTS
    # 368 VALUES PER DESCRIPTOR
    # 108 ORIENTATIONS PER OBJECT

    # import sample inventory data
    inventory = []
    i_file = "sampledata/assignment_3d_inventory_sample_data.txt"
    i_path = os.path.normpath(os.path.join(_HERE, i_file))
    with open(i_path, mode="r") as f:
        inventory = f.readlines()
    np_inventory = np.zeros((4, 108, 368))
    i = 0
    j = 0
    k = 0
    for x, val in enumerate(inventory):
        np_inventory[i, j, k] = val
        if j != 107 and k == 367:
            k = 0
            j += 1
            continue
        elif j == 107 and k == 367:
            i += 1
            j = 0
            k = 0
            continue
        k += 1

    # DATA FORMAT OF DESIGN SAMPLE DATA:
    # 2 OBJECTS
    # 368 VALUES PER DESCRIPTOR
    # 1 ORIENTATION PER OBJECT

    # import sample design data
    design = []
    d_file = "sampledata/assignment_3d_design_sample_data.txt"
    d_path = os.path.normpath(os.path.join(_HERE, d_file))
    with open(d_path, mode="r") as f:
        design = f.readlines()

    np_design = np.zeros((2, 368))
    i = 0
    k = 0
    for x, val in enumerate(design):
        np_design[i, k] = val
        if k == 367:
            k = 0
            i += 1
        else:
            k += 1

    # GENERATE 3D COST MATRIX FROM SAMPLE DATA
    # compare every design with all 108 orientations of all 4 objects and get
    # cost!

    # COST MATRIX STRUCTURE
    # i = 2
    # j = 4
    # k = 108
    cost = np.zeros((2, 4, 108))
    # loop over all design objects
    for i, d_obj in enumerate(np_design):
        # loop over all objects in the inventory per design object
        for j in range(np_inventory.shape[0]):
            # loop over orientations for every object in the inventory
            for k in range(np_inventory.shape[1]):
                pt1 = d_obj
                pt2 = np_inventory[j, k]
                cost_value = np.linalg.norm(pt2 - pt1, ord=2)
                cost[i, j, k] = cost_value

    # solve the sample data cost matrix
    solve_assignment_3d(cost, verbose=True)
