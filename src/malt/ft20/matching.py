# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import gurobipy as gp
import numpy as np

# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt import hopsutilities as hsutil


# FUNCTION DEFINITIONS --------------------------------------------------------

def optimize_matching(repository_components,
                      demand_components,
                      landfill_distances,
                      factory_distance,
                      transport_to_site,
                      reusecoeffs,
                      productioncoeffs,
                      verbose: bool = True):
    """
    Optimizes a matching for Fertigteil 2.0
    """
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
    # NOTE: cM should be a computed cost matrix, also depending on the demand!

    # BUILD NP ARRAYS FROM COMPONENTS -----------------------------------------

    # extract length and corss section from demand components
    demand_len = [obj.boundingbox[0] for obj in demand_components]
    demand_cs_x = [obj.boundingbox[1] for obj in demand_components]
    demand_cs_y = [obj.boundingbox[2] for obj in demand_components]

    # build 'm' matrix
    m = np.column_stack((np.array([round(x, 6) for x in demand_len]),
                         np.array([round(x, 6) for x in demand_cs_x]),
                         np.array([round(x, 6) for x in demand_cs_y])))

    # extract length and corss section from repository components
    stock_len = [obj.boundingbox[0] for obj in repository_components]
    stock_cs_x = [obj.boundingbox[1] for obj in repository_components]
    stock_cs_y = [obj.boundingbox[2] for obj in repository_components]

    # build 'R' matrix
    R = np.column_stack((np.array([round(x, 6) for x in stock_len]),
                         np.array([round(x, 6) for x in stock_cs_x]),
                         np.array([round(x, 6) for x in stock_cs_y])))

    # extract transport kilometers to lab
    # NOTE: assume that the first transport in history is always the one to lab
    transport_to_lab = [obj.transporthistory_parsed[0][2]
                        for obj in repository_components]

    # COMPOSE N ON BASIS OF m -------------------------------------------------

    cs_set = sorted(list(set([(x[1], x[2]) for x in m])), reverse=True)
    N = np.array([(float("inf"), x[0], x[1]) for x in cs_set])

    # initialize S as the size of R (stock) and N (new production)
    S = len(R) + len(N)

    # create assignment matrix T as an empty matrix
    T = np.zeros((len(m), S))

    print("[MIPHOPPER] Building Cost Array cS...")
    # cS -> cost array for disassembly and transport for every stock component
    cS = np.zeros(len(R))
    # loop over stock to compose cS
    for j, stock_obj in enumerate(R):
        # compute volume in m3
        volume = stock_obj[0] * stock_obj[1] * stock_obj[2]
        # compute individual impacts
        disassembly_impact = (
            volume *
            reusecoeffs['disassembly']
        )
        transport_lab_impact = (
            volume *
            reusecoeffs['transport_lab'] *
            transport_to_lab[j]
        )
        # set impact to cost array
        cS[j] = disassembly_impact + transport_lab_impact

    print("[MIPHOPPER] Building Cost Matrix cM...")
    # cM -> cost matrix for fabrication and installation of every component
    # regardless of new or reused!
    cM = np.zeros((len(m), S))
    # loop over demand to compose cM
    for i, demand_obj in enumerate(m):
        # compute volume in m3
        volume = demand_obj[0] * demand_obj[1] * demand_obj[2]
        # loop over stock + new production
        for j in range(S):
            if j < len(R):
                # if item is inside stock domain, compute environmental
                # impact as cost, based on reuse coefficients
                fabrication_impact = (
                    volume *
                    reusecoeffs['fabrication']
                )
                transport_site_impact = (
                    volume *
                    transport_to_site[j] *
                    reusecoeffs['transport_site']
                )
                assembly_impact = (
                    volume *
                    reusecoeffs['assembly']
                )
                # sum total impact of reuse
                total_reuse_impact = (
                    fabrication_impact +
                    transport_site_impact +
                    assembly_impact
                )
                # set impact to cost matrix
                cM[i, j] = total_reuse_impact
            else:
                # compute impacts based on volume
                demolition_impact = (
                    volume *
                    productioncoeffs['demolition']
                )
                # TODO: HOW TO CALC LANDFILL IMPACT FOR DEMAND ???
                # (NO ORIGINAL LOC) ???
                transport_landfill_impact = (
                    volume *
                    # landfill_distances[j] *
                    productioncoeffs['transport_landfill']
                )
                processing_impact = (
                    volume *
                    productioncoeffs['processing']
                )
                rawmat_manufacturing_impact = (
                    volume *
                    productioncoeffs['rawmat_manufacturing']
                )
                # NOTE: transport_rawmat_impact is NOT based on distance (!)
                transport_rawmat_impact = (
                    volume *
                    productioncoeffs['transport_rawmat']
                )
                fabrication_impact = (
                    volume *
                    productioncoeffs['fabrication']
                )
                transport_factory_impact = (
                    volume *
                    factory_distance *
                    productioncoeffs['transport_site']
                )
                assembly_impact = (
                    volume *
                    productioncoeffs['assembly']
                )
                # sum total production impact
                total_production_impact = (
                    demolition_impact +
                    transport_landfill_impact +
                    processing_impact +
                    rawmat_manufacturing_impact +
                    transport_rawmat_impact +
                    fabrication_impact +
                    transport_factory_impact +
                    assembly_impact
                )
                # set impact to cost matrix
                cM[i, j] = total_production_impact

    # print info and create profiler
    print("[MIPHOPPER] Building Gurobi Model for Matching Optimization...")
    timer = hsutil.Profiler()
    timer.start()

    # create the gurobi model
    model = gp.Model('FT2.0 Matching Optimization')

    # add binary decision variable if member i is either cut from stock element
    # {j ElementOf R} or produced new with cross-section {j ElementOf N}
    # NOTE: i / shape[0] is the row index (demand),
    #       j / shape[1] is the column index (stock)!
    t = model.addVars(T.shape[0],
                      T.shape[1],
                      vtype=gp.GRB.BINARY,
                      name='t')

    # add binary decision variable if one or more members are cut out from
    # stock element {j ElementOf R} (1) or if no member is cut out from stock
    # element {j ElementOf R} (0), i.e. element j remains unused.
    y = model.addVars(len(R),
                      vtype=gp.GRB.BINARY,
                      name='y')

    # for each member i, either one stock element j is reused or one new
    # element j is produced, as defined by the following constraint:
    # sum_{j = 1}^{s} t_{ij} = 1 for all i
    model.addConstrs((gp.quicksum(t[i, j] for j in range(S)) == 1
                      for i in range(T.shape[0])),
                     name='reuse_or_new')

    # the use of stock element {j ElementOf R} for one or more members is
    # constrained by the available length
    # NOTE:
    # m[i][0] = demand length (l´i)
    # R[j][0] = stock length (lj)
    model.addConstrs((
        gp.quicksum(t[i, j] * m[i][0] for i in range(len(m))) <= y[j] * R[j][0]
        for j in range(R.shape[0])),
        name='available_length'
    )

    # the assignment of members to elements is constrained by matching
    # cross sections
    # NOTE: m[i][1] = long cs demand
    #       RN[j][1] = long cs stock + production
    #       m[i][2] = short cs demand
    #       RN[j][2] = short cs stock + production
    # concatenate R and N to get all elements available from stock and new
    # production
    RN = np.concatenate((R, N), axis=0)
    for i in range(len(m)):
        for j in range(S):
            model.addConstr(
                t[i, j] * m[i][1] == t[i, j] * RN[j][1],
                name='cross_section_x')
            model.addConstr(
                t[i, j] * m[i][2] == t[i, j] * RN[j][2],
                name='cross_section_y')

    # the objective value is the sum of two cost indices
    # cS_j is the cost to source and process stock element {j ElementOf R}
    # cM_i,j is the cost to manufacture or install element j (reuse or new)
    # at position i
    # TODO: add correct cost values / computation
    #
    # NOTE:
    # - Production impact of 1m³ of concrete is the basis for caclulating
    # the cost of a new element
    # - "cj" is a placeholder for cS_j!
    # - cS is a cost list of the processing cost of the stock elements

    model.setObjective(
        gp.quicksum((cS[j] * y[j]) for j in range(len(R))) +

        gp.quicksum(cM[i, j] * t[i, j]
                    for i in range(T.shape[0])
                    for j in range(T.shape[1])),

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

    # Collect results of binary variable t
    # Reuse or New Production
    t_result = [(k[0], k[1]) for k in t.keys() if t[k].x > 0]
    y_result = [y[k].x for k in y.keys()]

    # Print some info
    # y = 1 if one or more members is cut from stock!
    if verbose:
        print(y.keys())
        print(y_result)

        [print("Demand {0}: Stock {1}".format(result[0], result[1]))
         for result in t_result]

    # return the optimal solution
    return (np.array(t_result), N)
