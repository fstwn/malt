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

    # add constraint so that each inventory object j can be used by exactly
    # one design object i
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
