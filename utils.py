import pandas as pd
from cpmpy import *
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("active_learning.log"),  
        logging.StreamHandler()  
    ]
)

def extract_features(constraint_info, dimension_sizes, problem_type):
    vars_in_scope = constraint_info[2]
    indices_list = [parse_variable_indices(var.name, dimension_sizes) for var in vars_in_scope]

    num_dimensions = len(dimension_sizes)
    average_positions = [0] * num_dimensions
    for indices in indices_list:
        for i in range(num_dimensions):
            average_positions[i] += indices[i]
    average_positions = [pos / len(vars_in_scope) for pos in average_positions]

    feature_dict = {
        'constraint_type': constraint_info[0],
        'problem_type': problem_type,
        'scope_size': len(vars_in_scope),
    }

    if num_dimensions == 2:
        rows = [indices[0] for indices in indices_list]
        cols = [indices[1] for indices in indices_list]

        feature_dict['is_full_row'] = int(len(set(rows)) == 1 and len(vars_in_scope) == dimension_sizes[0])
        feature_dict['is_full_column'] = int(len(set(cols)) == 1 and len(vars_in_scope) == dimension_sizes[1])
        feature_dict['is_full_block'] = int(is_full_block(vars_in_scope, dimension_sizes))
        feature_dict['is_diagonal'] = int(is_diagonal(vars_in_scope, dimension_sizes))

        feature_dict['average_row_position'] = sum(rows) / len(rows)
        feature_dict['average_column_position'] = sum(cols) / len(cols)

    return feature_dict

def prepare_features(constraint_info, dimension_sizes, problem_type):

    features = extract_features(constraint_info, dimension_sizes, problem_type)

    if features is None:
        logging.error("Feature extraction returned None. Skipping this constraint.")
        return None

    features_df = pd.DataFrame([features])

    categorical_features = ['constraint_type', 'problem_type']
    for cat_col in categorical_features:
        if cat_col in features_df.columns:
            features_df[cat_col] = features_df[cat_col].astype(str)
        else:
            features_df[cat_col] = 'Unknown'
            logging.warning(f"Missing categorical feature '{cat_col}'. Assigning default value 'Unknown'.")

    numerical_features = ['is_full_row', 'is_full_column', 'is_full_block',
                          'is_diagonal', 'average_row_position', 'average_column_position', 'scope_size']
    for num_col in numerical_features:
        if num_col not in features_df.columns:
            features_df[num_col] = 0
            logging.warning(f"Missing numerical feature '{num_col}'. Assigning default value 0.")

    features_df = features_df[['constraint_type', 'problem_type', 'is_full_row', 'is_full_column',
                               'is_full_block', 'is_diagonal', 'average_row_position',
                               'average_column_position', 'scope_size']]

    return features_df

def parse_model_file(file_path):
    max_index = -1
    constraints = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            constraint_type, indices_part = parts[0], parts[1]
            indices_part = indices_part.replace('[0]', '')
            indices = re.findall(r'\d+', indices_part)
            indices = [int(i) for i in indices]

            max_index_in_line = max(indices) if indices else -1
            if max_index_in_line > max_index:
                max_index = max_index_in_line

            if constraint_type == 'ALLDIFFERENT':
                constraints.append((constraint_type, indices))
            elif constraint_type == 'COUNT':
                if len(parts) >= 4:
                    count_value = int(parts[2])
                    count_times = int(parts[3])
                    constraints.append((constraint_type, indices, count_value, count_times))
                else:
                    print(f"Invalid COUNT constraint format in line: {line}")
            elif constraint_type == 'SUM':
                if len(parts) >= 3:
                    total_sum = int(parts[2])
                    constraints.append((constraint_type, indices, total_sum))
                else:
                    print(f"Invalid SUM constraint format in line: {line}")
            else:
                print(f"Unknown constraint type: {constraint_type} when parsing _model")

    return constraints, max_index


def construct_custom(experiment, data_dir="data/exp", oracle=None):
    import numpy as np
    import math
    import os
    import cpmpy as cp

    vars_file = f"{data_dir}/{experiment}_var"
    if not os.path.isfile(vars_file):
        raise FileNotFoundError(f"{vars_file} does not exist on experiment {experiment}.")
    vars_indices = parse_vars_file(vars_file)

    dom_file = f"{data_dir}/{experiment}_dom"
    domain_constraints = parse_dom_file(dom_file)
    lb, ub = domain_constraints[0][0], domain_constraints[0][1]

    num_vars = len(vars_indices)
    n = int(math.sqrt(num_vars))
    if n * n == num_vars:
        variables = cp.intvar(lb, ub, shape=(n, n), name="var")
        for i in range(n):
            for j in range(n):
                variables[i, j].name = f"var[{i},{j}]"
        grid = variables  # already 2D
        flat_vars_mapping = {i * n + j: variables[i, j] for i in range(n) for j in range(n)}
        flat_vars = [flat_vars_mapping[i] for i in range(n * n)]
    else:
        if oracle is not None and hasattr(oracle, 'variables_list'):
            if hasattr(oracle.variables_list, "shape") and len(oracle.variables_list.shape) == 2:
                rows, cols = oracle.variables_list.shape
            else:
                rows = int(math.floor(math.sqrt(num_vars)))
                cols = int(math.ceil(num_vars / rows))
        else:
            rows = int(math.floor(math.sqrt(num_vars)))
            cols = int(math.ceil(num_vars / rows))

        print(f"Warning: Number of variables ({num_vars}) is not a perfect square. "
              f"Using grid dimensions {rows} x {cols}.")

        variables = [cp.intvar(lb, ub, name=f"var[{i},{j}]") for i in range(rows) for j in range(cols)]
        if len(variables) > num_vars:
            variables = variables[:num_vars]
        grid = cp.cpm_array(np.array(variables).reshape((rows, cols)))
        flat_vars = variables
        flat_vars_mapping = { idx: var for idx, var in enumerate(variables) }

    model = cp.Model()
    con_file = f"{data_dir}/{experiment}_model"
    global_constraints = []
    if os.path.isfile(con_file):
        parsed_constraints, _ = parse_model_file(con_file)
        for constraint_data in parsed_constraints:
            constraint_type = constraint_data[0]
            if constraint_type == 'ALLDIFFERENT':
                _, indices = constraint_data
                vars_in_scope = [flat_vars_mapping[idx] for idx in indices if idx in flat_vars_mapping]
                constraint = cp.AllDifferent(vars_in_scope)
                model += constraint
                global_constraints.append(('ALLDIFFERENT', constraint, vars_in_scope))
            elif constraint_type == 'COUNT':
                _, indices, count_value, count_times = constraint_data
                vars_in_scope = [flat_vars_mapping[idx] for idx in indices if idx in flat_vars_mapping]
                constraint = sum([var == count_value for var in vars_in_scope]) == count_times
                model += constraint
                global_constraints.append(('COUNT', constraint, vars_in_scope, count_value, count_times))
            elif constraint_type == 'SUM':
                _, indices, total_sum = constraint_data
                vars_in_scope = [flat_vars_mapping[idx] for idx in indices if idx in flat_vars_mapping]
                constraint = sum(vars_in_scope) == total_sum
                model += constraint
                global_constraints.append(('SUM', constraint, vars_in_scope, total_sum))
            else:
                print(f"Unknown constraint type: {constraint_type}")
    else:
        print(f"{con_file} does not exist.")

    bias_file = f"{data_dir}/{experiment}_bias"
    biases = []
    if os.path.isfile(bias_file):
        with open(bias_file, 'r') as f:
            biases = [line.strip() for line in f if line.strip()]
    else:
        print(f"{bias_file} does not exist.")

    cl_file = f"{data_dir}/{experiment}_cl"
    cls = []
    if os.path.isfile(cl_file):
        with open(cl_file, 'r') as f:
            cls = [line.strip() for line in f if line.strip()]
    else:
        print(f"{cl_file} does not exist.")

    return grid, model, variables, biases, cls, global_constraints, flat_vars


def parse_variable_indices(var_name, dimension_sizes=None):

    if var_name.startswith("var["):
        match = re.match(r'var\[(.*?)\]', var_name)
        if match:
            indices_str = match.group(1)
            indices = tuple(map(int, indices_str.split(',')))
            return indices
        else:
            raise ValueError(f"Invalid variable name format: {var_name}")
    elif var_name.startswith("var"):
        idx = int(var_name[3:])
        if dimension_sizes is None:
            return (idx,)
        else:
            multi_indices = []
            remaining = idx
            for size in reversed(dimension_sizes):
                multi_indices.insert(0, remaining % size)
                remaining //= size
            return tuple(multi_indices)
    else:
        return (int(var_name),)

def multidimensional_indices_to_1d(indices, dimension_sizes):

    idx = 0
    for i, ind in enumerate(indices):
        multiplier = 1
        for size in dimension_sizes[i+1:]:
            multiplier *= size
        idx += ind * multiplier
    return idx

def flatten_vars(vars_obj):
    if isinstance(vars_obj, np.ndarray):
        return vars_obj.flatten().tolist()
    return vars_obj



def constraint_type_to_string(con_type):
    """
    Maps a numeric operator code to its string representation.

    Args:
        con_type (int): Operator code.

    Returns:
        str: The corresponding operator as a string.
    """
    mapping = {
        0: "!=",
        1: "==",
        2: ">",
        3: "<",
        4: ">=",
        5: "<="
    }
    return mapping.get(con_type, "Unknown")


def transform_bias_constraints(biases, vars_flatten):
    decomposed_bias_constraints = []
    pattern = re.compile(r'\(var(\d+)\)\s*([!=<>]+)\s*\(var(\d+)\)')

    for bias in biases:
        if isinstance(bias, str):
            tokens = bias.strip().split()
            if len(tokens) != 3:
                logging.warning(f"Bias string does not have exactly 3 tokens: '{bias}'. Skipping.")
                continue
            try:
                left_index = int(tokens[1])
                op_code = int(tokens[0])
                right_index = int(tokens[2])
            except ValueError:
                logging.warning(f"Could not parse bias tokens as integers in: '{bias}'. Skipping.")
                continue

            operator = constraint_type_to_string(op_code)
            if operator == "Unknown":
                logging.warning(f"Operator code {op_code} unknown in bias: '{bias}'. Skipping.")
                continue

            if not (0 <= left_index < len(vars_flatten)) or not (0 <= right_index < len(vars_flatten)):
                logging.warning(f"Indices out of range in bias: '{bias}'. Skipping.")
                continue

            var1 = vars_flatten[left_index]
            var2 = vars_flatten[right_index]

            if operator == "==":
                constraint = var1 == var2
            elif operator == "!=":
                constraint = var1 != var2
            elif operator == "<":
                constraint = var1 < var2
            elif operator == ">":
                constraint = var1 > var2
            elif operator == "<=":
                constraint = var1 <= var2
            elif operator == ">=":
                constraint = var1 >= var2
            else:
                logging.warning(f"Operator '{operator}' not recognized in bias: '{bias}'. Skipping.")
                continue

            decomposed_bias_constraints.append(constraint)

        elif isinstance(bias, Comparison):
            bias_str = str(bias)
            match = pattern.match(bias_str)
            if not match:
                logging.warning(f"Could not parse bias Comparison: '{bias}'. Skipping.")
                continue
            try:
                left_index = int(match.group(1))
                op_str = match.group(2)
                right_index = int(match.group(3))
            except ValueError:
                logging.warning(f"Error parsing bias Comparison: '{bias}'. Skipping.")
                continue
            if not (0 <= left_index < len(vars_flatten)) or not (0 <= right_index < len(vars_flatten)):
                logging.warning(f"Indices out of range in bias Comparison: '{bias}'. Skipping.")
                continue
            var1 = vars_flatten[left_index]
            var2 = vars_flatten[right_index]
            operator = op_str  # Already a string such as "==", "!=" etc.
            if operator == "==":
                constraint = var1 == var2
            elif operator == "!=":
                constraint = var1 != var2
            elif operator == "<":
                constraint = var1 < var2
            elif operator == ">":
                constraint = var1 > var2
            elif operator == "<=":
                constraint = var1 <= var2
            elif operator == ">=":
                constraint = var1 >= var2
            else:
                logging.warning(f"Operator '{operator}' not recognized in bias Comparison: '{bias}'. Skipping.")
                continue

            decomposed_bias_constraints.append(constraint)
        else:
            logging.warning(f"Unexpected bias type: {bias} (type {type(bias)}). Skipping.")
            continue

    return decomposed_bias_constraints


def transform_bias_constraints_pl_mapping(biases, instance_mapping, instance):

    dims = list(instance.variables.shape)
    constraints = []
    for bias in biases:
        tokens = bias.strip().split()
        if len(tokens) != 3:
            continue
        var1_key, operator, var2_key = tokens
        if var1_key in instance_mapping and var2_key in instance_mapping:
            var1 = instance_mapping[var1_key]
            var2 = instance_mapping[var2_key]
            tup1 = parse_variable_indices(var1.name)
            tup2 = parse_variable_indices(var2.name)
            index1 = multidimensional_indices_to_1d(tup1, dims)
            index2 = multidimensional_indices_to_1d(tup2, dims)
            var1 =  instance.variables.flatten().tolist()[index1]
            var2 = instance.variables.flatten().tolist()[index2]

            if operator == "!=":
                constraints.append(var1 != var2)
            elif operator == "==":
                constraints.append(var1 == var2)
            elif operator == "<":
                constraints.append(var1 < var2)
            elif operator == ">":
                constraints.append(var1 > var2)
            elif operator == "<=":
                constraints.append(var1 <= var2)
            elif operator == ">=":
                constraints.append(var1 >= var2)
            else:
                continue
        else:
            print(f"Warning: One or both variable keys '{var1_key}', '{var2_key}' not found in instance mapping.")
    return constraints


import math
import shutil
import time
from cpmpy import intvar, boolvar, Model, all, sum, SolverLookup
from sklearn.utils import class_weight
import numpy as np
import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain, combinations
import networkx as nx
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import cpmpy as cp


def build_instance_mapping(variables):

    mapping = {}
    max_row = -1
    max_col = -1
    pattern_var = re.compile(r"var\[(\d+),\s*(\d+)\]")
    pattern_x = re.compile(r"x(\d+)_(\d+)")

    for var in variables:
        if var.name.startswith("var["):
            m = pattern_var.match(var.name)
            if not m:
                raise ValueError(f"Variable name '{var.name}' does not match the expected format 'var[i,j]'.")
            row = int(m.group(1))
            col = int(m.group(2))
            key = f"x{row + 1}_{col + 1}"
        elif var.name.startswith("x"):
            m = pattern_x.match(var.name)
            if not m:
                raise ValueError(f"Variable name '{var.name}' does not match the expected format 'x<number>_<number>'.")
            row = int(m.group(1)) - 1
            col = int(m.group(2)) - 1
            key = var
        else:
            raise ValueError(f"Variable name '{var.name}' does not match expected formats.")
        mapping[key] = var
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    grid_dims = (max_row + 1, max_col + 1)
    return mapping, grid_dims

def parse_learned_constraint(constraint_str, instance_mapping):
    try:
        start = constraint_str.index("(")
        end = constraint_str.index(")")
    except ValueError:
        return None
    var_names = constraint_str[start + 1:end].split(", ")
    vars_in_scope = [instance_mapping[vn] for vn in var_names if vn in instance_mapping]
    if not vars_in_scope:
        return None
    if constraint_str.startswith("AllDifferent") or constraint_str.startswith("ExamDayAllDifferent"):
        constraint_obj = cp.AllDifferent(vars_in_scope)
        return ("ALLDIFFERENT", constraint_obj, vars_in_scope)
    elif constraint_str.startswith("Sum"):
        return ("SUM", constraint_str, vars_in_scope)
    elif constraint_str.startswith("Count"):
        return ("COUNT", constraint_str, vars_in_scope)
    else:
        return None


def save_valid_constraints(global_constraints, invalid_constraints, output_file_path):

    with open(output_file_path, 'w') as f:
        for gc in global_constraints:
            key = tuple(gc[:2] + (tuple(var.name for var in gc[2]),))

            if key in invalid_constraints:
                continue

            constraint_type = gc[0]
            vars_in_scope = [var.name for var in gc[2]]

            if constraint_type == 'ALLDIFFERENT':
                line = f"ALLDIFFERENT\t{' '.join(vars_in_scope)}\n"
            elif constraint_type == 'SUM':
                if len(gc) >= 4:
                    total_sum = gc[3]
                    line = f"SUM\t{' '.join(vars_in_scope)}\t{total_sum}\n"
                else:
                    continue
            elif constraint_type == 'COUNT':
                if len(gc) >= 5:
                    count_value = gc[3]
                    count_times = gc[4]
                    line = f"COUNT\t{' '.join(vars_in_scope)}\t{count_value}\t{count_times}\n"
                else:
                    continue
            else:
                continue

            f.write(line)


def parse_model_file_vars(model_file_path):
    variable_indices = set()
    with open(model_file_path, 'r') as file:
        for line in file:
            indices = re.findall(r'\[(.*?)\]', line)
            for group in indices:
                nums = re.findall(r'\d+', group)
                variable_indices.update(map(int, nums))
    return variable_indices


def is_full_row(vars_in_scope, dimension_sizes=[9, 9]):

    indices = []
    for var in vars_in_scope:
        if '[' in var.name:
            tup = parse_variable_indices(var.name)
            flat = multidimensional_indices_to_1d(tup, dimension_sizes)
        else:
            flat = int(var.name[3:])
        indices.append(flat)
    n = dimension_sizes[0]
    if len(indices) != n:
        return False
    rows = [idx // n for idx in indices]
    if len(set(rows)) != 1:
        return False
    cols = [idx % n for idx in indices]
    return set(cols) == set(range(n))


def is_full_column(vars_in_scope, dimension_sizes=[9, 9]):
    indices = []
    for var in vars_in_scope:
        if '[' in var.name:
            tup = parse_variable_indices(var.name)
            flat = multidimensional_indices_to_1d(tup, dimension_sizes)
        else:
            flat = int(var.name[3:])
        indices.append(flat)
    n = dimension_sizes[0]
    if len(indices) != n:
        return False
    cols = [idx % n for idx in indices]
    if len(set(cols)) != 1:
        return False
    rows = [idx // n for idx in indices]
    return set(rows) == set(range(n))


def is_full_block(vars_in_scope, dimension_sizes=[9, 9]):

    indices = []
    for var in vars_in_scope:
        if '[' in var.name:
            tup = parse_variable_indices(var.name)
            flat = multidimensional_indices_to_1d(tup, dimension_sizes)
        else:
            flat = int(var.name[3:])
        indices.append(flat)
    n = dimension_sizes[0]
    rows = [idx // n for idx in indices]
    cols = [idx % n for idx in indices]
    block_rows = [r // 3 for r in rows]
    block_cols = [c // 3 for c in cols]
    if len(set(block_rows)) != 1 or len(set(block_cols)) != 1:
        return False
    br = block_rows[0]
    bc = block_cols[0]
    expected_indices = [ (r * n + c)
                         for r in range(br * 3, br * 3 + 3)
                         for c in range(bc * 3, bc * 3 + 3) ]
    return set(indices) == set(expected_indices)


def is_diagonal(vars_in_scope, dimension_sizes=[9, 9]):

    indices = []
    for var in vars_in_scope:
        if '[' in var.name:
            tup = parse_variable_indices(var.name)
            flat = multidimensional_indices_to_1d(tup, dimension_sizes)
        else:
            flat = int(var.name[3:])
        indices.append(flat)
    n = dimension_sizes[0]
    if len(indices) != n:
        return False
    main_diagonal = [i * n + i for i in range(n)]
    anti_diagonal = [i * n + (n - 1 - i) for i in range(n)]
    return set(indices) == set(main_diagonal) or set(indices) == set(anti_diagonal)

def parse_dom_file(file_path):
    domain_constraints = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3:
                var_index = int(parts[0])
                lower_bound = int(parts[2])
                upper_bound = int(parts[-1])
                domain_constraints[var_index] = (lower_bound, upper_bound)
    return domain_constraints

def parse_con_file(file_path):
    biases = []

    with open(file_path, 'r') as file:
        for line in file:
            con_type, var1, var2 = map(int, line.strip().split())
            biases.append((con_type, var1, var2))

    return biases


def parse_vars_file(file_path):
    with open(file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        vars_values = [0] * total_vars

        for i, line in enumerate(file):
            value, _ = map(int, line.split())
            vars_values[i] = value

    return vars_values


def write_solutions_to_json(instance, size, format_template, solutions, non_solutions, problem_type, output_file):
    data = {
        "instance": instance,
        "size": size,
        "formatTemplate": format_template,
        "solutions": [{"array": sol} for sol in solutions],
        "nonSolutions": [{"array": non_sol} for non_sol in non_solutions],
        "problemType": problem_type
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)