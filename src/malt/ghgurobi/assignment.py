# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import gurobipy as gp
import numpy as np

# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import hopsutilities as hsutil


# FUNCTION DEFINITIONS --------------------------------------------------------

def solve_assignment(cost: np.array):
    """
    Solves an assignment problem defined by the given cost matrix using Gurobi.
    The cost matrix does not need to be square, but num of columns
    (tasks/inventory) has to be larger or equal to num of rows
    (workers/design).
    """

    # BUILD MODEL -------------------------------------------------------------

    # print info and create profiler
    print("[GHGUROBI] Building Gurobi Model...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("Assignment")

    # add binary decision variables for num of rows and num of cols of cost matrix
    # NOTE: shape[0] is the row index, shape[1] is the column index!
    x = model.addVars(cost.shape[0],
                      cost.shape[1],
                      vtype=gp.GRB.BINARY,
                      name="x")
    model.update()

    # create mapping dict from gurobi variable names to values
    # TODO: this is super hacky, improve it and save computation time!
    mapping = {(x[i, j].VarName): [i, j]
               for i in range(cost.shape[0])
               for j in range(cost.shape[1])}

    # add constraint so that each worker i can perform at maximum one task j
    # sum_{i = 1}^{n} x_{ij} <= 1 for all j
    model.addConstrs((gp.quicksum(x[i, j] for i in range(cost.shape[0])) <= 1
                     for j in range(cost.shape[1])), name="workload")

    # add constraint so that each task j is performed by exactly one worker i
    # sum_{j = 1}^{n} x_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(x[i, j] for j in range(cost.shape[1])) == 1
                     for i in range(cost.shape[0])), name="completion")

    # set objective function as sum of cost times the decision variable
    # sum_{i = 1}^{n} sum_{j = 1}^{n} c_{ij} x_{ij}
    model.setObjective(gp.quicksum(cost[i, j] * x[i, j]
                                   for i in range(cost.shape[0])
                                   for j in range(cost.shape[1])),
                       gp.GRB.MINIMIZE)

    # update the model to add constraints and objective function
    model.update()

    # don't print all of the info...
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
    for v in model.getVars():
        if v.x > 0:
            key = v.VarName
            assignment_result.append(mapping[key][1])
            assignment_cost.append(cost[mapping[key][0], mapping[key][1]])
            # print("%s: %g" % (key, v.x))

    # return the optimal solution
    return np.array(assignment_result), np.array(assignment_cost)


# TESTING ---------------------------------------------------------------------

if __name__ == "__main__":
    # solve a rectangular cost ,atrix using gurobi
    cost_matrix = np.random.uniform(1, 100, (10, 100))
    solve_assignment(cost_matrix)
