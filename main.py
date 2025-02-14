import os, math, random, re, time, pickle, logging
import numpy as np
import pandas as pd
import cpmpy as cp
from cpmpy import Model, intvar, AllDifferent
from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.get_variables import get_variables_model
import joblib
from utils import build_instance_mapping
from benchmarks import construct_sudoku, construct_examtt_simple, construct_nurse_rostering, \
        construct_jsudoku, construct_greaterThansudoku
from benchmarks_global import construct_jsudoku as cj_global
from benchmarks_global import construct_nurse_rostering as nr_global
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_sudoku as cs_global
from benchmarks_global import construct_greaterThansudoku as cgts_global

def parse_variable_indices(var_name, dimension_sizes=None):
    if var_name.startswith("var["):
        match = re.match(r'var\[(.*?)\]', var_name)
        if match:
            indices = tuple(map(int, match.group(1).split(',')))
            return indices
        else:
            raise ValueError(f"Invalid variable name format: {var_name}")
    else:
        return (int(var_name),)


def compute_variable_involvement(global_constraints):
    involvement = {}
    for gc in global_constraints:
        vars_in_scope = gc[2]
        for var in vars_in_scope:
            involvement[var.name] = involvement.get(var.name, 0) + 1
    return involvement

def score_pair(var1, var2, involvement, alpha=1, beta=1):
    try:
        r1, c1 = parse_variable_indices(var1.name)
        r2, c2 = parse_variable_indices(var2.name)
        manhattan = abs(r1 - r2) + abs(c1 - c2)
        involv = involvement.get(var1.name, 0) + involvement.get(var2.name, 0)
        return alpha * manhattan - beta * involv
    except Exception as e:
        return 0


def is_valid_solution(assignment, target_model, time_limit=5):
    temp_constraints = target_model.constraints[:]
    for var in target_model.variables_list:
        if var.name in assignment:
            temp_constraints.append(var == assignment[var.name])
    m_temp = Model(temp_constraints)
    solver = CPM_ortools(m_temp)
    solver.ort_solver.parameters.max_time_in_seconds = time_limit
    return solver.solve()


def print_assignment(assignment, variables, grid_size):
    grid = np.full((grid_size, grid_size), '.', dtype=object)
    for var in variables:
        try:
            i, j = parse_variable_indices(var.name)
            if var.name in assignment:
                grid[i, j] = str(assignment[var.name])
        except Exception as e:
            continue
    for row in grid:
        print(" ".join(row))


def update_probability_bayesian(key, feedback, constraint_probabilities):
    prior = constraint_probabilities.get(key, 0.4)
    if feedback:
        posterior = 0.0
    else:
        posterior = (0.75 * prior) / (0.75 * prior + 0.25 * (1 - prior))
    constraint_probabilities[key] = posterior


def query_driven_refinement(C_G, all_constraints, constraint_probabilities, theta_max, timeout, target_model):

    def candidate_key(candidate):
        return tuple(candidate[:2] + (tuple(var.name for var in candidate[2]),))

    print("Starting query-driven refinement with detailed logging...")
    start_time = time.time()
    total_violation_queries = 0
    sorted_candidates = sorted(C_G, key=lambda c: constraint_probabilities.get(candidate_key(c), 0.4))
    refined_candidates = []
    candidate_constraints = [cand[1] for cand in C_G]
    involvement = compute_variable_involvement(C_G)

    for candidate in sorted_candidates:
        key = candidate_key(candidate)
        print("\n-----------------------------------------------------")
        print(f"Processing candidate: {key}")
        S = candidate[2]
        print(f"Variables in candidate: {[var.name for var in S]}")

        query_count = 0
        candidate_accepted = False
        candidate_refuted = False

        while not candidate_accepted and not candidate_refuted:
            print(f"\nQuery attempt {query_count + 1} for candidate {key}:")
            total_violation_queries += 1
            violating_found = False
            violation_assignment = None
            var_pairs = []
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    dom_i = set(range(S[i].lb, S[i].ub + 1))
                    dom_j = set(range(S[j].lb, S[j].ub + 1))
                    print(
                        f"Pair ({S[i].name}, {S[j].name}): Domain_i = {sorted(list(dom_i))}, Domain_j = {sorted(list(dom_j))}")
                    if dom_i & dom_j:
                        inter = sorted(list(dom_i & dom_j))
                        s = score_pair(S[i], S[j], involvement)
                        var_pairs.append((s, S[i], S[j]))
                        print(f" -> Intersection found: {inter} with score {s}")
                    else:
                        print(" -> No intersection found.")
            if not var_pairs:
                print("No candidate pairs with intersecting domains found; candidate accepted without query.")
                candidate_accepted = True
                break

            var_pairs.sort(key=lambda tup: tup[0], reverse=True)
            print(f"Sorted pairs by score: {[(p[1].name, p[2].name, p[0]) for p in var_pairs]}")

            for score, var_i, var_j in var_pairs:
                dom_i = set(range(var_i.lb, var_i.ub + 1))
                dom_j = set(range(var_j.lb, var_j.ub + 1))
                inter = sorted(list(dom_i & dom_j))
                print(f"Trying pair ({var_i.name}, {var_j.name}) with intersection: {inter}")
                if inter:
                    v = random.choice(inter)
                    print(f"Selected forced value {v} for pair ({var_i.name}, {var_j.name}).")
                    new_constraints = [con for con in all_constraints if con not in candidate_constraints]
                    new_constraints += [var_i == v, var_j == v]
                    print("Building modified model (M') without candidate constraints and with forced assignments.")
                    m_prime = Model(new_constraints)
                    solver = CPM_ortools(m_prime)
                    solver.ort_solver.parameters.max_time_in_seconds = timeout
                    if solver.solve():
                        violation_vars = get_variables_model(m_prime)
                        violation_assignment = {var.name: var.value() for var in violation_vars}
                        print(f"Modified model solvable. Generated violation assignment: {violation_assignment}")
                        violating_found = True
                        break
                    else:
                        print("Modified model unsolvable with this forced assignment. Trying next pair.")
            if not violating_found:
                print("No violation assignment could be generated in this query attempt.")
                query_count += 1
                continue

            print(f"Sending query to oracle for candidate {key} with assignment: {violation_assignment}")
            feedback = is_valid_solution(violation_assignment, target_model)
            print(
                f"Oracle response: {'Valid solution (query accepted)' if feedback else 'Invalid solution (query refuted)'}")

            if feedback:
                old_prob = constraint_probabilities.get(key, 0.4)
                print(
                    f"Candidate {key} refuted by violation query; prior probability was {old_prob:.2f}. Setting new probability to 0.0.")
                constraint_probabilities[key] = 0.0
                candidate_refuted = True
            else:
                old_prob = constraint_probabilities.get(key, 0.4)
                update_probability_bayesian(key, feedback,
                                            constraint_probabilities=constraint_probabilities)
                new_prob = constraint_probabilities[key]
                print(f"Candidate {key} updated probability: {old_prob:.2f} -> {new_prob:.2f}")
                if new_prob >= theta_max:
                    print(f"Candidate {key} accepted with high confidence (p={new_prob:.2f}).")
                    candidate_accepted = True
                else:
                    print(f"Candidate {key} remains low (p={new_prob:.2f}). Continuing query...")
            query_count += 1

        if candidate_accepted:
            refined_candidates.append(candidate)
        else:
            print(f"Candidate {key} ultimately refuted after {query_count} queries.")
    end_time = time.time()
    violation_time = end_time - start_time
    print(
        f"Query-driven refinement completed. Total violation queries: {total_violation_queries}, total time: {violation_time:.2f} seconds.")
    return refined_candidates, total_violation_queries, violation_time

def construct_instance(experiment_name):
    if 'jsudoku' in experiment_name.lower():
        print("Constructing jsudoku")
        n = 9
        instance_binary, oracle_binary = construct_jsudoku()
        instance_global, oracle_global = cj_global()
    elif 'greaterthan' in experiment_name.lower():
        print("Constructing greaterthan")
        n = 9
        instance_binary, oracle_binary = construct_greaterThansudoku(3, 3, 9)
        instance_global, oracle_global = cgts_global(3, 3, 9)
    elif '4sudoku' in experiment_name.lower():
        print("Constructing 4sudoku")
        n = 4
        instance_binary, oracle_binary = construct_sudoku(2, 2, 4)
        instance_global, oracle_global = cs_global(2, 2, 4)
    elif '9sudoku' in experiment_name.lower():
        print("Constructing 9sudoku")
        n = 9
        instance_binary, oracle_binary = construct_sudoku(3, 3, 9)
        instance_global, oracle_global = cs_global(3, 3, 9)
    elif 'examtt' in experiment_name.lower():
        print("Constructing examtt")
        n = 6
        instance_binary, oracle_binary = construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9,
                                                                 days_for_exams=14)
        instance_global, oracle_global = ces_global(nsemesters=9, courses_per_semester=6, slots_per_day=9,
                                                    days_for_exams=14)
    elif 'nurse' in experiment_name.lower():
        print("Constructing nurse rostering")
        instance_binary, oracle_binary = construct_nurse_rostering()
        instance_global, oracle_global = nr_global()
        n = None
    else:
        print("Constructing 9sudoku")
        n = 9
        instance_binary, oracle_binary = construct_sudoku(3, 3, 9)
        instance_global, oracle_global = cs_global(3, 3, 9)
        raise ValueError("Unknown experiment name")
    return instance_binary, oracle_binary, instance_global, oracle_global, n

def decompose_constraints(final_constraints, instance, multidimensional_indices_to_1d):
    decomposed_constraints_set = []
    dims = list(instance.variables.shape)
    from itertools import combinations
    for gc in final_constraints:
        if gc[0] == 'ALLDIFFERENT':
            vars_in_scope = gc[2]
            for var1, var2 in combinations(vars_in_scope, 2):
                try:
                    tup1 = parse_variable_indices(var1.name)
                    tup2 = parse_variable_indices(var2.name)
                    index1 = multidimensional_indices_to_1d(tup1, dims)
                    index2 = multidimensional_indices_to_1d(tup2, dims)
                except Exception as e:
                    print(f"Error parsing variable names '{var1.name}' or '{var2.name}': {e}")
                    continue
                if not (0 <= index1 < len(instance.variables.flatten().tolist())) or not (0 <= index2 < len(instance.variables.flatten().tolist())):
                    continue
                var_flatten1 = instance.variables.flatten().tolist()[index1]
                var_flatten2 = instance.variables.flatten().tolist()[index2]
                constraint_tuple = var_flatten1 != var_flatten2
                decomposed_constraints_set.append(constraint_tuple)
    return decomposed_constraints_set

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from utils import multidimensional_indices_to_1d, transform_bias_constraints_pl_mapping

    experiment = "exps/9sudoku_solution.json"
    instance_binary, oracle_binary, instance, oracle_global, n = construct_instance(experiment)
    from passive import PassiveLearningSystem

    max_sols = 20
    data_dir = os.path.join("./bench", experiment)

    pl_system = PassiveLearningSystem(experiment, max_solutions=max_sols)
    pl_system.learn_constraints()
    global_constraints, model = pl_system.get_learned_constraints_and_model(instance.X)

    print(f"Learned candidate global constraints (passive) ({len(global_constraints)}):")
    for candidate in global_constraints:
        print(candidate)

    all_constraints = model.constraints

    clf = joblib.load("random_forest_constraint_classifier.pkl")
    with open('unified_X_train_columns.pkl', 'rb') as f:
        columns = pickle.load(f)


    def candidate_key(candidate):
        return tuple(candidate[:2] + (tuple(var.name for var in candidate[2]),))


    constraint_probabilities = {}
    for candidate in global_constraints:
        key = candidate_key(candidate)
        if candidate[0] != 'ALLDIFFERENT':
            constraint_probabilities[key] = 1.0
            continue
        from utils import prepare_features
        dimension_sizes = [n, n]
        features_df = prepare_features(candidate, dimension_sizes, "sudoku")
        prob = clf.predict_proba(features_df)[0][1]
        constraint_probabilities[key] = prob
        print(f"Candidate {key} ML probability: {prob:.2f}")

    theta_max = 0.9 # Confidence threshold.
    timeout = 5
    oracle_global.variables_list = instance.X

    refined_candidates, total_violation_queries, violation_time = query_driven_refinement(
        global_constraints, all_constraints, constraint_probabilities, theta_max, timeout, oracle_global, max_queries=3)
    print(f"\nRefined candidate constraints after query-driven refinement ({len(refined_candidates)}):")
    for candidate in refined_candidates:
        print(candidate)

    final_constraints = [c[1] for c in refined_candidates]
    model.constraints = final_constraints
    vars_flatten = instance.variables.flatten().tolist()

    from pycona import MQuAcq2

    ca_system = MQuAcq2()
    variables = instance.X
    mapping, grid_dims = build_instance_mapping(variables)
    instance.bias = transform_bias_constraints_pl_mapping(pl_system.bias, mapping, instance)
    instance.cl = decompose_constraints(refined_candidates,instance,multidimensional_indices_to_1d)


    print(f"Bias ({len(instance.bias )}:",instance.bias[:10])
    print(f"CL ({len(instance.cl )}:",instance.cl[:10])
    learned_instance = ca_system.learn(instance, oracle=oracle_global, verbose=3)
    print(ca_system.env.metrics.short_statistics)

    final_constraints_active = learned_instance.get_cpmpy_model().constraints

    results = {
        'Prob.': experiment.split("_solution_sol")[0] if "_solution_sol" in experiment else experiment,
        'Sols': experiment.split("_solution_sol")[1] if "_solution_sol" in experiment else max_sols,
        'StartC': len(global_constraints),
        'InvC': len(global_constraints) - len(refined_candidates),
        'FinC': len(refined_candidates),
        'ViolQ': total_violation_queries,
        'Bias': len(instance.bias),
        'MQuQ': ca_system.env.metrics.total_queries,
        'TQ': total_violation_queries + ca_system.env.metrics.total_queries,
        'VT(s)': round(violation_time, 2),
        'MQuT(s)': round(ca_system.env.metrics.total_time, 2),
        'TT(s)': round(violation_time + ca_system.env.metrics.total_time, 2)
    }
    results_df = pd.DataFrame([results])
    csv_file = 'results.csv'
    print(results_df)
    if os.path.exists(csv_file):
        results_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_file, mode='w', header=True, index=False)
