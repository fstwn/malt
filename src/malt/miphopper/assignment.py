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
    print("[MIPHOPPER] Building Gurobi Model...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("2D Assignment Problem")

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
    print("[MIPHOPPER] Building model took {0} ms".format(timer.rawstop()))

    # optimize the model and time it with the simple profiler
    timer.start()
    model.optimize()

    # stop profiler and print time elaspsed for solving
    print("[MIPHOPPER] Solving model took {0} ms".format(timer.rawstop()))

    # collect the results of the assignment
    asm_result, asm_cost = zip(*[(sk[1], cost[sk[0], sk[1]])
                                 for sk in select.keys() if select[sk].x > 0])

    if verbose:
        [print("Design {0}: Iventory {1} // Cost: {2}".format(i, r, c))
         for i, (r, c) in enumerate(zip(asm_result, asm_cost))]

    # return the optimal solution
    return np.array(asm_result), np.array(asm_cost)


def solve_assignment_3d(cost: np.array, verbose: bool = False):
    """
    Solves an assignment problem defined by a given 3d cost matrix using
    Gurobi.
    """

    # print info and create profiler
    print("[MIPHOPPER] Building Gurobi Model...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("3D Assignment Problem")

    # add binary decision variables
    select = model.addVars(cost.shape[0],
                           cost.shape[1],
                           cost.shape[2],
                           vtype=gp.GRB.BINARY,
                           name="select")

    # add constraint so that each design i can be filled by at maximum one
    # inventory object j
    # sum_{i = 1}^{n} x_{ij} <= 1 for all j
    model.addConstrs((gp.quicksum(select[i, j, k] for i in range(cost.shape[0])
                      for k in range(cost.shape[2])) <= 1
                     for j in range(cost.shape[1])), name="select_workload")

    # add constraint so that each inventory object j can be used by exactly
    # one design object i
    # sum_{j = 1}^{n} x_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(select[i, j, k] for j in range(cost.shape[1])
                      for k in range(cost.shape[2])) == 1
                     for i in range(cost.shape[0])), name="select_completion")

    # set objective function as sum of cost times the decision variable
    # sum_{i = 1}^{n} sum_{j = 1}^{n} c_{ij} x_{ij}
    model.setObjective(gp.quicksum(cost[i, j, k] * select[i, j, k]
                                   for i in range(cost.shape[0])
                                   for j in range(cost.shape[1])
                                   for k in range(cost.shape[2])),
                       gp.GRB.MINIMIZE)

    # don't print all of the info...
    if not verbose:
        model.setParam("OutputFlag", False)

    # stop the profiler and print time elapsed for building model
    print("[MIPHOPPER] Building model took {0} ms".format(timer.rawstop()))

    # optimize the model and time it with the simple profiler
    timer.start()
    model.optimize()

    # stop profiler and print time elaspsed for solving
    print("[MIPHOPPER] Solving model took {0} ms".format(timer.rawstop()))

    # collect the results of the assignment
    asm_result, asm_cost = zip(*[((sk[1], sk[2]), cost[sk[0], sk[1], sk[2]])
                                 for sk in select.keys() if select[sk].x > 0])

    if verbose:
        [print("Design {0}: Iventory {1} -> Orientation {2} // "
               "Cost: {3}".format(i, r[0], r[1], c))
         for i, (r, c) in enumerate(zip(asm_result, asm_cost))]

    # return the optimal solution
    return np.array(asm_result), np.array(asm_cost)
