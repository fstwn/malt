# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os


# ADDITIONAL MODULE IMPORTS ---------------------------------------------------

import numpy as np


# LOCAL MODULE IMPORTS --------------------------------------------------------

import malt


# TEST DEFINITIONS ------------------------------------------------------------

def test_solve_assignment_2d():
    fn = "assignment_2d_sample_data.txt"
    cost_matrix = np.loadtxt(os.path.normpath(os.path.join(malt.DATADIR, fn)))
    asm, cst = malt.ghgurobi.solve_assignment_2d(cost_matrix, verbose=True)
    assert asm[0] == 34
    assert asm[1] == 43
    assert asm[2] == 60
    assert asm[3] == 52
    assert asm[4] == 45
    assert asm[5] == 48
    assert asm[6] == 78
    assert asm[7] == 55
    assert asm[8] == 80
    assert asm[9] == 70


def test_solve_assignment_3d():
    # DATA FORMAT OF DESIGN SAMPLE DATA:
    # 2 OBJECTS
    # 1 ORIENTATION PER OBJECT
    # 368 VALUES PER DESCRIPTOR

    # import sample design data
    design = []
    d_file = "assignment_3d_design_sample_data.txt"
    d_path = os.path.normpath(os.path.join(malt.DATADIR, d_file))
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

    # DATA FORMAT OF INVENTORY SAMPLE DATA:
    # 4 OBJECTS
    # 108 ORIENTATIONS PER OBJECT
    # 368 VALUES PER DESCRIPTOR

    # import sample inventory data
    inventory = []
    i_file = "assignment_3d_inventory_sample_data.txt"
    i_path = os.path.normpath(os.path.join(malt.DATADIR, i_file))
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
    asm, cst = malt.ghgurobi.solve_assignment_3d(cost, verbose=True)

    assert cst[0] == min(cost[0, asm[0][0]])
    assert cst[1] == min(cost[1, asm[1][0]])
    assert asm[0][0] == 1
    assert asm[0][1] == 68
    assert asm[1][0] == 3
    assert asm[1][1] == 5
