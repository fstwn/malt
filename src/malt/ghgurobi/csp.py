# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import gurobipy as gp
import numpy as np

# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import hopsutilities as hsutil


# FUNCTION DEFINITIONS --------------------------------------------------------

def solve_csp(m: np.array,
              R: np.array,
              N: np.array,
              verbose: bool = True):
    """
    Solves a cutting stock problem using Gurobi.
    """

    # !!! STOCK HAS TO BE DENOTED BY J INDEX !!!

    # m = demand of members to be built, defined by cross-section and length
    # R = stock of reclaimed elements for reuse, predefined cross-section and
    #     length
    # N = new production, predefined cross-section but length is "infinite"
    # S = size of stock R + size of the set of elements available from new
    #     production N

    # Assignment matrix {T ElementOf {0, 1} ^ m x s}
    # T is a matrix with *m* as rows and *S* as columns
    # in the beginning this matrix should be just zeroes

    # cS_j is the cost to source and process stock element {j ElementOf R}
    # cS is an array of all source/processing costs per member {j ElementOf R}

    # cM_i,j is the cost to manufacture or install element j (reuse or new) at
    # position i
    # cM should be a computed cost matrix

    # -------------------------------------------------------------------------

    # initialize S as the size of R (stock) and N (new production)
    S = len(R) + len(N)
    # create assignment matrix T as an empty matrix
    T = np.zeros((len(m), S))

    print("[GHGUROBI] Building Cost Matrix cM...")

    # TODO: find better cost quantification!
    # cM -> should be the cost matrix,
    # but not necessarily difference in length!
    cM = np.zeros((len(m), S))
    # loop over demand
    for i, sobj in enumerate(m):
        for j in range(S):
            # cross-check with stock + new production
            if j < len(R):
                # if item is inside stock domain, use absolute length
                # difference as cost (??)
                cM[i, j] = abs(R[j][0] - sobj[0])
            else:
                # otherwise use arbitrary high number
                cM[i, j] = 9999999

    # print info and create profiler
    print("[GHGUROBI] Building Gurobi Model for Cutting Stock Problem...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model("Cutting Stock Problem")

    # add binary decision variable if member i is either cut from stock element
    # {j ElementOf R} or produced new with cross-section {j ElementOf N}
    # NOTE: i / shape[0] is the row index (demand),
    #       j / shape[1] is the column index (stock)!
    t = model.addVars(T.shape[0],
                      T.shape[1],
                      vtype=gp.GRB.BINARY,
                      name="t")

    # add binary decision variable if one or more members are cut out from
    # stock element {j ElementOf R} (1) or if no member is cut out from stock
    # element {j ElementOf R} (0), i.e. element j remains unused.
    y = model.addVars(len(R),
                      vtype=gp.GRB.BINARY,
                      name="y")

    # for each member i, either one stock element j is reused or one new
    # element j is produced, as defined by the following constraint:
    # sum_{j = 1}^{s} t_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(t[i, j] for j in range(S)) == 1
                      for i in range(T.shape[0])),
                     name="reuse_or_new")

    # the use of stock element {j ElementOf R} for one or more members is
    # constrained by the available length
    # NOTE:
    # m[i][0] = demand length (l´i)
    # R[j][0] = stock length (lj)
    model.addConstrs((gp.quicksum(t[i, j] * m[i][0] for i in range(len(m))) <= y[j] * R[j][0] # NOQA501
                     for j in range(R.shape[0])),
                     name="available_length")

    # concatenate R and N to get all elements available from stock and new
    # production
    RN = np.concatenate((R, N), axis=0)
    # the assignment of members to elements is constrained by matching
    # cross sections
    # NOTE: m[i][1] = long cs demand
    #       RN[j][1] = long cs stock + production
    #       m[i][2] = short cs demand
    #       RN[j][2] = short cs stock + production
    for i in range(len(m)):
        for j in range(S):
            model.addConstr(t[i, j] * m[i][1] == t[i, j] * RN[j][1])
            model.addConstr(t[i, j] * m[i][2] == t[i, j] * RN[j][2])

    # the objective value is the sum of two cost indices
    # cS_j is the cost to source and process stock element {j ElementOf R}

    # cM_i,j is the cost to manufacture or install element j (reuse or new)
    # at position i
    # TODO: add correct cost values / computation
    cj = 10
    model.setObjective(
        gp.quicksum((cj * y[j]) for j in range(len(R))) +

        gp.quicksum(cM[i, j] * t[i, j]
                    for i in range(T.shape[0])
                    for j in range(T.shape[1])),

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

    # collect the results of the optimisation
    t_result = [(k[0], k[1]) for k in t.keys() if t[k].x > 0]

    # y_result = [y[k].x for k in y.keys()]

    # Print some info
    if verbose:
        [print("Demand {0}: Stock {1}".format(result[0], result[1]))
         for result in t_result]

    # return the optimal solution
    return np.array(t_result)
