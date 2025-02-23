import cpmpy as cp

from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_greaterThansudoku(block_size_row, block_size_col, grid_size):
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
        model += cp.AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += cp.AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]).decompose()  # python's indexing

    greater_than_pairs = [
        ((3, 3), (3, 4)),  # m += (x[3, 3] > x[3, 4])
        ((0, 2), (0, 3)),  # m += (x[0, 2] > x[0, 3])
        ((1, 7), (1, 8)),  # m += (x[1, 7] > x[1, 8])
        ((2, 3), (2, 4)),  # m += (x[2, 3] > x[2, 4])
        ((4, 0), (4, 1)),  # m += (x[4, 0] > x[4, 1])
        ((4, 3), (4, 4)),  # m += (x[4, 3] > x[4, 4])
        ((6, 7), (6, 8)),  # m += (x[6, 7] > x[6, 8])
    ]
    for (x1, y1), (x2, y2) in greater_than_pairs:
        model += grid[x1, y1] > grid[x2, y2]

    C_T = list(model.constraints)

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="construct_greaterThansudoku")

    oracle = ConstraintOracle(C_T)

    print(len(C_T))
    return instance, oracle
