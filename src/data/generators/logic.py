# a && a == a
# a || a == a
# !T == F
# !F == T
# T && T == T
# T && F == F
# F && T == F
# F && F == F
# T || T == T
# T || F == T
# F || T == T
# F || F == F
# a && T == a
# a || F == a
# lit && F == F
# lit || T == T
# !(!a || !b) == a && b
# !(!a && !b) == a || b
import random
import re
import string
from data.vocabulary import SEP, HAL

# Arg < TrueArg, FalseArg, LitArg
# expansion_rules: Op or LogicExpressions
# __repr__

class Arg:
    def __init__(self):
        self.expansion_rules = None

    def __repr__(self):
        pass


class TrueArg(Arg):

    def __init__(self):
        self.value = True

    def __repr__(self):
        return "T"

    def expand(self):
        expansion_rules = [
            NotOp([FalseArg()]),
            OrOp([LitArg(), TrueArg()]),
            AndOp([TrueArg(), TrueArg()]),
            OrOp([TrueArg(), TrueArg()]),
            OrOp([TrueArg(), FalseArg()]),
            OrOp([FalseArg(), TrueArg()]),
        ]
        return random.choice(expansion_rules)


class FalseArg(Arg):

    def __init__(self):
        self.value = False

    def __repr__(self):
        return "F"

    def expand(self):
        expansion_rules = [
            NotOp([TrueArg()]),
            AndOp([LitArg(), FalseArg()]),
            AndOp([TrueArg(), FalseArg()]),
            AndOp([FalseArg(), TrueArg()]),
            AndOp([FalseArg(), FalseArg()]),
            OrOp([FalseArg(), FalseArg()]),
        ]
        return random.choice(expansion_rules)


class LitArg(Arg):

    def __init__(self, variable=None):
        if variable is None:
            self.variable = random.choice('abcdefghijklmnopqrstuvwxyz')
        else:
            self.variable = variable

    def __repr__(self):
        return str(self.variable)

    def expand(self):
        expansion_rules = [
            AndOp([LitArg(self.variable), TrueArg()]),
            OrOp([LitArg(self.variable), FalseArg()]),
            AndOp([LitArg(self.variable), LitArg(self.variable)]),
            OrOp([LitArg(self.variable), LitArg(self.variable)])
        ]
        return random.choice(expansion_rules)

    @property
    def value(self):
        return self.variable


# Op < AndOp, OrOp, NotOp
# - arguments
# - nesting_level (if in LogicExpression)
# - value (compression_rule)
# - __repr__
# - truth_value


class Op:

    def __init__(self, args):
        self.args = args
        self.depth = 1
        self.paths_to_expansion_points = [[arg_pos] for arg_pos in range(len(args))]
        self.max_expansion_points = len(args)
        # self.solution_steps = [(self, str(self), str(self.value))]
        self.solution_steps = {str(self): str(self.value)}

    def __repr__(self):
        pass

    def _expand(self, expansion_path):
        # print("Expanding:", self, "Path:", expansion_path)

        if len(expansion_path) == 1:
            expansion_point = expansion_path[0]
            expansion = self.args[expansion_point].expand()
            self.args[expansion_point] = expansion
            self.paths_to_expansion_points.remove(expansion_path)
            self.paths_to_expansion_points += [[expansion_path[0], arg_pos] for arg_pos in range(len(self.args[expansion_point].args))]
            return self
        else:
            # if (isinstance(self, AndOp) or isinstance(self, OrOp)) and random.random() < 0.5:
            #     expansion = self.expand_demorgan()
            expansion_point = expansion_path[0]
            self.args[expansion_point]._expand(expansion_path[1:])
            self.paths_to_expansion_points.remove(expansion_path)
            self.paths_to_expansion_points += [[expansion_path[0]] + path for path in self.args[expansion_point].paths_to_expansion_points]
            return self

    def expand(self):
        if len(self.paths_to_expansion_points) > 1:
            num_samples = 2
        else:
            num_samples = 1

        max_len_paths = set([tuple(path) for path in self.paths_to_expansion_points if len(path) == self.depth])
        expansion_paths = random.sample(max_len_paths, num_samples)
        expansion_paths = [list(path) for path in expansion_paths]

        # tmp_steps = []
        for path in expansion_paths:
            self._expand(path)
            sub_expression = self.get_subexpression_from_path(path)
            self.solution_steps |= {str(sub_expression): str(sub_expression.value)}
            # tmp_steps.append((sub_expression, str(sub_expression), str(sub_expression.value)))
        # self.solution_steps += tmp_steps[::-1]
        self.depth += 1

    def get_subexpression_from_path(self, path):
        current = self.args[path[0]]
        for p in path[1:]:
            current = current.args[p]
        return current



class AndOp(Op):

    def __init__(self, args):
        super().__init__(args)

    @property
    def value(self):
        for a in self.args:
            if isinstance(a, FalseArg):
                return FalseArg()

        if isinstance(self.args[0], TrueArg) and isinstance(self.args[1], TrueArg):
            return TrueArg()

        if isinstance(self.args[0], LitArg) and isinstance(self.args[1], LitArg) and (self.args[0].value == self.args[1].value):
            return self.args[0]

        if (isinstance(self.args[0], LitArg) and isinstance(self.args[1], FalseArg) or
                (isinstance(self.args[1], LitArg) and isinstance(self.args[0], FalseArg))):
            return FalseArg()

        if isinstance(self.args[0], LitArg) and isinstance(self.args[1], TrueArg):
            return self.args[0]

        if isinstance(self.args[1], LitArg) and isinstance(self.args[0], TrueArg):
            return self.args[1]

    def expand_demorgan(self):
        # !(!a || !b)
        return NotOp([OrOp([NotOp([self.args[0]]), NotOp([self.args[1]])])])

    def __repr__(self):
        return "(" + str(self.args[0]) + "&" + str(self.args[1]) + ")"


class OrOp(Op):

    def __init__(self, args):
        super().__init__(args)


    @property
    def value(self):
        for a in self.args:
            if isinstance(a, TrueArg):
                return TrueArg()

        if isinstance(self.args[0], FalseArg) and isinstance(self.args[1], FalseArg):
            return FalseArg()

        if isinstance(self.args[0], LitArg) and isinstance(self.args[1], LitArg) and (self.args[0].value == self.args[1].value):
            return self.args[0]

        if ((isinstance(self.args[0], LitArg) and isinstance(self.args[1], TrueArg)) or
                (isinstance(self.args[1], LitArg) and isinstance(self.args[0], TrueArg))):
            return TrueArg()

        if isinstance(self.args[0], LitArg) and isinstance(self.args[1], FalseArg):
            return self.args[0]

        if isinstance(self.args[1], LitArg) and isinstance(self.args[0], FalseArg):
            return self.args[1]

    def expand_demorgan(self):
        # !(!a && !b)
        # (!((!F) & (!F)) == (F | F)
        return NotOp([AndOp([NotOp([self.args[0]]), NotOp([self.args[1]])])])

    def __repr__(self):
        return "(" + str(self.args[0]) + "|" + str(self.args[1]) + ")"


class NotOp(Op):

    def __init__(self, args):
        super().__init__(args)

    @property
    def value(self):
        if isinstance(self.args[0], TrueArg):
            return FalseArg()

        elif isinstance(self.args[0], FalseArg):
            return TrueArg()

        elif isinstance(self.args, OrOp):
            pass

        elif isinstance(self.args, AndOp):
            pass

    def __repr__(self):
        return "(!" + str(self.args[0]) + ")"

class LogicExpression:

    def __init__(self):
        self.expr = None

    def build(self, nesting):
        start_arg = random.choice([TrueArg(), FalseArg(), LitArg()])
        if nesting == 0:
            self.expr = start_arg
            self.solution_chain = [start_arg]
            self.sub_expr = [start_arg]
            self.steps = [(None, str(start_arg), HAL)]
            return self
        expression = start_arg.expand()
        for _ in range(nesting-1):
            expression.expand()
        self.expr = expression
        self.simplify()
        return self

    def __repr__(self):
        return self.expr.__repr__()

    def simplify(self):
        leaf_expr_re = re.compile('\([a-zTF][|&][a-zTF]\)|\(![a-zTF]\)')
        expr = str(self.expr)
        self.solution_chain = [expr]
        self.sub_expr = []
        self.steps = []

        while len(expr) > 1:
            sub_expr = leaf_expr_re.findall(expr)[-1]
            value = self.expr.solution_steps[sub_expr]
            expr = expr.replace(sub_expr, value, 1)
            self.solution_chain += [expr]
            self.sub_expr += [sub_expr]
            self.steps += [(None, sub_expr, value)]

    def print_solution_chain(self):
        for step in self.solution_chain:
            print(step)

    def get_solution_chain_stats(self):
        return [(self._compute_depth_expression_str(x), 2, x, y) for x, y in zip(self.solution_chain, self.sub_expr)]

    def _compute_depth_expression_str(self, expression_string):
        depth, max_depth = 0, 0
        for c in expression_string:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth > max_depth:
                max_depth = depth
        return max_depth


class LogicExpressionGenerator:

    def __init__(self):
        self.vocab_chars = string.ascii_lowercase + 'TF&|!()'


    def generate_samples(self, num_samples, nesting, num_operands, split, exact, task):
        # wrapper for compatibility
        return self._generate_samples(num_samples, nesting, task, split)

    def _generate_samples(self, num_samples, nesting, task, split='train'):
        samples = [self.generate_sample(nesting, split) for _ in range(num_samples)]
        self.samples = samples

        X_simplify_w_value, Y_simplify_w_value = self._build_simplify_w_value(samples)
        Y_solve = self._build_solve_target(samples)
        Y_select = self._build_select_target(samples)
        Y_solve_atomic = self._build_solve_atomic_target(samples)

        X_by_task = {
            'select': X_simplify_w_value,
            'solve_atomic': X_simplify_w_value,
            'solve': X_simplify_w_value,
        }

        Y_by_task = {
            'select': Y_select,
            'solve_atomic': Y_solve_atomic,
            'solve': Y_solve,
        }

        if (nesting == 2):
            X_by_task['select_s1'] = self._build_select_step_input(samples, 's1')
            Y_by_task['select_s1'] = self._build_select_step_target(samples, 's1')

        if (nesting == 3):
            for step in ['s1', 's2', 's3', 's4']:
                X_by_task[f'select_{step}'] = self._build_select_step_input(samples, step)
                Y_by_task[f'select_{step}'] = self._build_select_step_target(samples, step)

        inputs = []
        targets = []

        for task_name in task:
            if not task_name in Y_by_task:
                assert False, f"Wrong task name: {task_name}."
            else:
                inputs.append(X_by_task[task_name])
                targets.append(Y_by_task[task_name])

        if len(targets) == 1:
            inputs = inputs[0]
            targets = targets[0]

        return inputs, targets

    def generate_sample(self, nesting, split):
        if split is None:
            return self._generate_sample_no_split(nesting)
        else:
            return self._generate_sample_in_split(split, nesting)

    def _generate_sample_in_split(self, split, nesting):
        current_split = ''

        while current_split != split:
            expression = self._generate_sample_no_split(nesting)
            sample_hash = hash(str(expression))

            if sample_hash % 3 == 0:
                current_split = 'train'

            elif sample_hash % 3 == 1:
                current_split = 'valid'

            else:
                current_split = 'test'
        return expression

    def _generate_sample_no_split(self, nesting):
        expression = LogicExpression()
        expression.build(nesting)
        return expression

    def _build_simplify_w_value(self, samples):
        X_str = []
        Y_str = []

        for sample in samples:
            X_str.append(str(sample))

            if sample.steps[0][2] == HAL:
                Y_str.append(f"{HAL}")

            else:
                Y_str.append(f"{sample.steps[0][1]}{SEP}{sample.steps[0][2]}")

        return X_str, Y_str


    def _build_select_target(self, samples):
        Y_str = []

        for sample in samples:

            if sample.steps[0][2] == HAL:
                Y_str.append(str(sample.steps[-1][1]))
            else:
                Y_str.append(f"{sample.steps[0][1]}")

        return Y_str

    def _build_select_step_input(self, samples, step):
        X_str = []
        step_idx = int(step[1])

        for sample in samples:
            if len(sample.solution_chain) <= step_idx+1:  # avoid problems with unary NOT op
                continue
            X_str.append(sample.solution_chain[step_idx])

        return X_str

    def _build_select_step_target(self, samples, step):
        Y_str = []
        step_idx = int(step[1])

        for sample in samples:
            if len(sample.sub_expr) <= step_idx:  # avoid problems with unary NOT op
                continue
            Y_str.append(sample.sub_expr[step_idx])

        return Y_str

    def _build_solve_target(self, samples):
        return [str(sample.steps[-1][2]) for sample in samples]

    def _build_solve_atomic_target(self, samples):
        Y_str = []

        for sample in samples:
            if sample.steps[0][2] == HAL:
                Y_str.append(f"{HAL}")
            else:
                Y_str.append(str(sample.steps[-1][2]))

        return Y_str


if __name__ == '__main__':
    start_arg = random.choice([TrueArg(), FalseArg(), LitArg()])
    print(start_arg, 0)
    f = start_arg.expand()
    print(f, f.depth)
    f.expand()
    print(f, f.depth)
    f.expand()
    print(f, f.depth)
    f.expand()
    print(f, f.depth)
    f.expand()
    print(f, f.depth)
    f.expand()
    print(f, f.depth)
    print(f.solution_steps)
