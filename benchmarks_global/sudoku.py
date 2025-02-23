import cpmpy as cp

from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_sudoku(block_size_row, block_size_col, grid_size):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Create a dictionary with the parameters
    parameters = {"block_size_row": block_size_row, "block_size_col": block_size_col, "grid_size": grid_size}

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="var")

    model = cp.Model()

    # Constraints on rows and columns
    for row in grid:
        model += cp.AllDifferent(row)

    for col in grid.T:  # numpy's Transpose
        model += cp.AllDifferent(col)   

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col])

    C_T = list(model.constraints)

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="sudoku")

    oracle = ConstraintOracle(C_T)

    print(len(C_T))
    return instance, oracle
