import json
import random
from cpmpy import Model, AllDifferent

from utils import parse_learned_constraint, build_instance_mapping


class PassiveLearningSystem:
    def __init__(self, solutions_file, max_solutions):
        with open(solutions_file, 'r') as file:
            data = json.load(file)
        self.params = data.get("params", {})
        self.solutions = data['solutions'][:max_solutions]
        self.variables = self.extract_variables(self.solutions)
        self.grid_size = len(self.solutions[0]['array'])
        self.block_size = int(self.grid_size ** 0.5)
        self.window_size = self.block_size
        self.domain = self.calculate_domain()
        self.bias = self.generate_bias()
        self.learned_global_constraints = []
        print(f"Passive learning with {len(self.solutions)} solutions.")

    def extract_variables(self, solutions):
        first_solution = solutions[0]['array']
        variables = []
        for i in range(len(first_solution)):
            for j in range(len(first_solution[i])):
                variables.append(f'x{i+1}_{j+1}')
        return variables

    def calculate_domain(self):
        domain = set()
        for solution in self.solutions:
            for row in solution['array']:
                domain.update(row)
        return sorted(domain)

    def generate_bias(self):
        bias = []
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                var1 = self.variables[i]
                var2 = self.variables[j]
                bias.extend([
                    f'{var1} != {var2}',
                    f'{var1} == {var2}',
                    f'{var1} > {var2}',
                    f'{var1} < {var2}',
                    f'{var1} >= {var2}',
                    f'{var1} <= {var2}'
                ])
        return bias

    def learn_constraints(self):
        candidate_constraints = self.identify_candidate_constraints()
        for constraint in candidate_constraints:
            if self.validate_constraint(constraint):
                self.learned_global_constraints.append(constraint)
        self.filter_fixed_arity_constraints()

    def identify_candidate_constraints(self):
        candidate_constraints = set()
        all_vars = [f'x{i+1}_{j+1}' for i in range(self.grid_size) for j in range(self.grid_size)]
        candidate_constraints.add(f'AllDifferent({", ".join(all_vars)})')

        for i in range(self.grid_size):
            vars_row = [f'x{i+1}_{j+1}' for j in range(self.grid_size)]
            candidate_constraints.add(f'AllDifferent({", ".join(vars_row)})')

        for i in range(self.grid_size):
            vars_row = [f'x{i+1}_{j+1}' for j in range(self.grid_size)]
            candidate_constraints.add(f'AllDifferent({", ".join(vars_row)})')

        for j in range(self.grid_size):
            vars_col = [f'x{i+1}_{j+1}' for i in range(self.grid_size)]
            candidate_constraints.add(f'AllDifferent({", ".join(vars_col)})')
        for i in range(0, self.grid_size, self.block_size):
            for j in range(0, self.grid_size, self.block_size):
                vars_block = []
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        vars_block.append(f'x{i+x+1}_{j+y+1}')
                candidate_constraints.add(f'AllDifferent({", ".join(vars_block)})')
        diag_main = [f'x{i+1}_{i+1}' for i in range(self.grid_size)]
        candidate_constraints.add(f'AllDifferent({", ".join(diag_main)})')
        diag_anti = [f'x{i+1}_{self.grid_size-i}' for i in range(self.grid_size)]
        candidate_constraints.add(f'AllDifferent({", ".join(diag_anti)})')
        for i in range(self.grid_size - self.window_size + 1):
            for j in range(self.grid_size - self.window_size + 1):
                vars_window = []
                for x in range(self.window_size):
                    for y in range(self.window_size):
                        vars_window.append(f'x{i+x+1}_{j+y+1}')
                candidate_constraints.add(f'AllDifferent({", ".join(vars_window)})')
        num_random_selections = 5
        for _ in range(num_random_selections):
            vars_rand = random.sample(self.variables, k=4)
            candidate_constraints.add(f'AllDifferent({", ".join(vars_rand)})')
            for i in range(self.grid_size - self.window_size + 1):
                for j in range(self.grid_size - self.window_size + 1):
                    vars_window = []
                    for x in range(self.window_size):
                        for y in range(self.window_size):
                            vars_window.append(f'x{i + x + 1}_{j + y + 1}')
                    candidate_constraints.add(f'AllDifferent({", ".join(vars_window)})')
            for window in range(2, self.grid_size // 2 + 1):
                for i in range(self.grid_size - window + 1):
                    for j in range(self.grid_size - window + 1):
                        vars_window = []
                        for x in range(window):
                            for y in range(window):
                                vars_window.append(f'x{i + x + 1}_{j + y + 1}')
                        candidate_constraints.add(f'AllDifferent({", ".join(vars_window)})')

            num_random_selections = 5
            for _ in range(num_random_selections):
                vars_rand = random.sample(self.variables, k=4)
                candidate_constraints.add(f'AllDifferent({", ".join(vars_rand)})')


            return list(candidate_constraints)
        for i in range(self.grid_size):
            vars_row = [f'x{i+1}_{j+1}' for j in range(self.grid_size)]
            self.add_sum_constraints(candidate_constraints, vars_row)
        for j in range(self.grid_size):
            vars_col = [f'x{i+1}_{j+1}' for i in range(self.grid_size)]
            self.add_count_constraints(candidate_constraints, vars_col)
        for i in range(0, self.grid_size, self.block_size):
            for j in range(0, self.grid_size, self.block_size):
                vars_block = []
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        vars_block.append(f'x{i+x+1}_{j+y+1}')
                self.add_sum_constraints(candidate_constraints, vars_block)
                self.add_count_constraints(candidate_constraints, vars_block)
        return list(candidate_constraints)

    def add_sum_constraints(self, candidate_constraints, variables):
        min_sum = len(variables) * min(self.domain)
        max_sum = len(variables) * max(self.domain)
        for target_sum in range(min_sum, max_sum + 1):
            candidate_constraints.add(f'Sum({", ".join(variables)}) == {target_sum}')

    def add_count_constraints(self, candidate_constraints, variables):
        for value in self.domain:
            for target_count in range(len(variables) + 1):
                candidate_constraints.add(f'Count({", ".join(variables)}, {value}) == {target_count}')

    def validate_constraint(self, constraint):
        for solution in self.solutions:
            model = self.create_model(solution, constraint)
            if not model.solve():
                return False
        return True

    def create_model(self, solution, constraint):
        model = Model()
        variables = {}
        arr = solution['array']
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                var_name = f'x{i+1}_{j+1}'
                variables[var_name] = int(arr[i][j])
        if constraint.startswith('AllDifferent'):

            vars_in_constraint = constraint.split('(')[1].split(')')[0].split(', ')
            valid_vars = [var for var in vars_in_constraint if var in variables]
            if len(valid_vars) < len(vars_in_constraint):
                print(
                    f"Warning: In constraint '{constraint}', some variables are missing. Using valid variables only: {valid_vars}")
            model += AllDifferent([variables[var] for var in valid_vars])
        elif constraint.startswith('AllDifferent'):
            vars_in_constraint = constraint.split('(')[1].split(')')[0].split(', ')
            slots_per_day = self.params.get("slots_per_day", 9)  # Default if not provided.
            exam_day_consts = [ (variables[var] - 1) // slots_per_day for var in vars_in_constraint ]
            model += AllDifferent(exam_day_consts)
        elif constraint.startswith('Sum'):
            vars_in_constraint = constraint.split('(')[1].split(')')[0].split(', ')
            target_sum = int(constraint.split('==')[1].strip())
            model += sum([variables[var] for var in vars_in_constraint]) == target_sum
        elif constraint.startswith('Count'):
            inside = constraint.split('(')[1].split(')')[0]
            parts = inside.split(',')
            target_value = int(parts[-1].strip())
            target_count = int(constraint.split('==')[1].strip())
            vars_in_constraint = [p.strip() for p in parts[:-1]]
            model += sum([1 if variables[var] == target_value else 0 for var in vars_in_constraint]) == target_count
        return model

    def filter_fixed_arity_constraints(self):
        new_bias = []
        for constraint in self.bias:
            if self.is_constraint_satisfied(constraint):
                new_bias.append(constraint)
        self.bias = new_bias

    def is_constraint_satisfied(self, constraint):
        for solution in self.solutions:
            if not self.check_constraint_in_solution(solution, constraint):
                return False
        return True

    def check_constraint_in_solution(self, solution, constraint):
        # For binary constraints in the bias.
        var1, op, var2 = constraint.split()
        val1 = int(solution['array'][int(var1[1]) - 1][int(var1[3]) - 1])
        val2 = int(solution['array'][int(var2[1]) - 1][int(var2[3]) - 1])
        if op == '!=':
            return val1 != val2
        elif op == '==':
            return val1 == val2
        elif op == '>':
            return val1 > val2
        elif op == '<':
            return val1 < val2
        elif op == '>=':
            return val1 >= val2
        elif op == '<=':
            return val1 <= val2
        return False

    def print_learned_constraints(self):
        print(f"Learned Global Constraints ({len(self.learned_global_constraints)}):")
        for constraint in self.learned_global_constraints:
            print(constraint)
        print(f"\nRemaining Bias ({len(self.bias)}):")
        for constraint in self.bias:
            print(constraint)

    def get_learned_constraints_and_model(self, variables):
        mapping, grid_dims = build_instance_mapping(variables)
        learned_tuples = []
        for constr_str in self.learned_global_constraints:
            parsed = parse_learned_constraint(constr_str, mapping)
            if parsed is not None:
                learned_tuples.append(parsed)
            else:
                print(f"Error parsing constraint: {constr_str}")
        constraints_list = [c for (ctype, c, vs) in learned_tuples]
        model = Model(constraints_list)
        return learned_tuples, model
