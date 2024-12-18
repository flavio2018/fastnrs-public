import logging
import re
import string

import torch
from models.new_combiner import Combiner
from models.new_finder import Finder
from models.selsolcom import SelSolComBatchRunHistory, SelSolCom, SelectorOutput


class EncSelSolCom(SelSolCom):

    def __init__(self, selector_encoder, solver, vocabulary, n_multi=1):
        super(EncSelSolCom, self).__init__(
            selector_encoder, solver, vocabulary, n_multi=n_multi
        )
        self.selector_encoder = selector_encoder
        self.solver = solver
        self.solver.vocabulary.sos = False
        self.finder = Finder(vocabulary)
        self.combiner = Combiner()
        self.vocabulary = vocabulary

    def forward(self, expression):
        batch_size = expression.size(0)
        self.run_batch_history = SelSolComBatchRunHistory(batch_size)
        self.count_solver_errors_per_seq = torch.zeros(batch_size).to(expression.device)
        self.track_stop_reason = {
            "sub_expression": torch.zeros(batch_size).to(expression.device),
            "parentheses": torch.zeros(batch_size).to(expression.device),
            "no_valid_leaf": torch.zeros(batch_size).to(expression.device),
        }
        is_valid = torch.ones(batch_size).bool().to(expression.device)
        is_final = torch.zeros(batch_size).bool().to(expression.device)

        expression_str = self.vocabulary.batch_to_str(expression[is_valid])
        task_name = self.solver.vocabulary.task_name_from_sample(expression_str[0])
        if (
            "listops" in self.vocabulary.tokenizer
            or "arithmetic" in self.vocabulary.tokenizer
            or "algebra" in self.vocabulary.tokenizer
        ):
            sample_atomic_value = "1"
        elif "char" in self.vocabulary.tokenizer:
            sample_atomic_value = "T"

        prev_leafs_batch = None
        while torch.bitwise_and(is_valid, ~is_final).any():
            expression_str = self.vocabulary.batch_to_str(expression[is_valid])

            # handling case alltask
            self.current_task = self.vocabulary.task_name_from_sample(expression_str[0])
            if "listops" in self.current_task:
                sample_atomic_value = "1"
                confidence_score_threshold = -6
            elif "arithmetic" in self.current_task:
                sample_atomic_value = "1"
                confidence_score_threshold = -2
            elif "algebra" in self.current_task:
                sample_atomic_value = "1"
                confidence_score_threshold = -3
            elif "logic" in self.current_task:
                sample_atomic_value = "T"
                confidence_score_threshold = -0.005

            self.run_batch_history.update_selector_inputs(expression_str, is_valid)

            leafs_batch = self.textseg_encoder_fwd(expression[is_valid])
            if prev_leafs_batch is not None and leafs_batch == prev_leafs_batch:
                logging.info("Same leaves two times, breaking loop...")
                break
            prev_leafs_batch = leafs_batch
            max_num_leafs = max(len(leafs) for leafs in leafs_batch)
            leafs_full_batch = []
            idx = 0
            for idx_batch, is_val in enumerate(is_valid):
                if is_val:
                    leafs_full_batch.append(leafs_batch[idx])
                    # check it's actually a valid output (ie not all None)
                    if sum(int(leaf is None) for leaf in leafs_batch[idx]) == len(
                        leafs_batch
                    ):
                        is_valid[idx_batch] = False
                        self.track_stop_reason["no_valid_leaf"][idx_batch] += 1
                        logging.info("[ERRLOG][SOLVER] no_valid_leafs!")
                    idx += 1
                else:
                    leafs_full_batch.append([None] * max_num_leafs)
            leafs_batch = leafs_full_batch

            all_low_quality = torch.zeros(batch_size).bool().to(expression.device)
            for leaf_idx in range(max_num_leafs):
                sub_expression_str = [
                    leafs[leaf_idx]
                    for leafs, is_val in zip(leafs_batch, is_valid)
                    if is_val
                ]
                # trick the solver
                mod_sub_expr_str = [
                    se if se is not None else sample_atomic_value
                    for se in sub_expression_str
                ]
                for idx, leaf in enumerate(mod_sub_expr_str):
                    if not self._subexpr_has_well_formed_parentheses(leaf):
                        logging.info(
                            f"[ERRLOG][SELECTOR] Found leaf expression with malformed parentheses: {leaf}"
                        )
                sub_expression = self.solver.vocabulary.str_to_batch(mod_sub_expr_str)
                sub_expression = self.get_full_batch(
                    sub_expression,
                    is_valid,
                    self.solver.vocabulary.get_special_idx("pad"),
                )
                solution_tmp_obj = self._solver_multi_run_hq(sub_expression[is_valid])

                is_low_quality = []
                valid_solution_tmp_str = []
                possible_error_due_to_selector = [False] * len(is_valid)
                idx = 0
                for idx_full_batch, is_val in enumerate(is_valid):
                    if not is_val:
                        continue
                    solver_output = solution_tmp_obj[idx]
                    valid_solution_tmp_str.append(solver_output.text)
                    if solver_output.confidence_score < confidence_score_threshold:
                        logging.info(
                            f"[ERRLOG][SELECTOR] Low confidence Input: {mod_sub_expr_str[idx]} | {solver_output}"
                        )
                        is_low_quality.append(True)
                        all_low_quality[idx_full_batch] = False
                    else:
                        is_low_quality.append(False)
                        if (sub_expression_str[idx] is not None) and not (
                            self._subexpr_has_well_formed_parentheses(
                                sub_expression_str[idx]
                            )
                        ):
                            possible_error_due_to_selector[idx_full_batch] = True

                            # is_valid[idx_full_batch] = False
                            # self.track_stop_reason["parentheses"][idx_full_batch] += 1
                    idx += 1

                # trick the solver
                valid_solution_tmp_str = [
                    sol if se is not None else sample_atomic_value
                    for sol, se in zip(valid_solution_tmp_str, sub_expression_str)
                ]
                solution = self.solver.vocabulary.str_to_batch(
                    valid_solution_tmp_str, x=False
                )
                # self.run_batch_history.update_solver_outputs(solution_tmp_str, is_valid)

                solution = self.get_full_batch(
                    solution,
                    is_valid,
                    self.solver.vocabulary.get_special_idx("pad", x=False),
                )

                # update the final sequences based on the output of the solver
                is_final = self._update_is_final(is_final, solution)
                solution_str = self.solver.vocabulary.batch_to_str(
                    solution[is_valid], x=False
                )
                solution_str = [
                    s.replace(self.solver.vocabulary.specials["eos"], "")
                    for s in solution_str
                ]
                expression_str = self.vocabulary.batch_to_str(expression[is_valid])

                upd_expression_str = []
                valid_solver_outs = []
                val_counter = 0
                for is_val in is_valid:
                    if is_val:
                        e_str, leaf, sol_str, is_lq = (
                            expression_str[val_counter],
                            sub_expression_str[val_counter],
                            solution_str[val_counter],
                            is_low_quality[val_counter],
                        )
                        if (leaf is not None) and (sol_str != "$") and not is_lq:
                            upd_expression_str.append(e_str.replace(leaf, sol_str))
                            valid_solver_outs.append(1)
                        else:
                            upd_expression_str.append(e_str)
                            valid_solver_outs.append(0)
                        val_counter += 1
                    else:
                        upd_expression_str.append("")
                        valid_solver_outs.append(0)

                logging.info(f"Updated expression string: {upd_expression_str}")
                # count errors due to solver mistakes
                valid_solver_outs = (
                    torch.tensor(valid_solver_outs).bool().to(expression.device)
                )
                if valid_solver_outs.any():
                    pred_is_wrong = self.update_solver_errors_counter(
                        solution, sub_expression, valid_solver_outs
                    )
                    idx = 0
                    for idx_batch, is_val in enumerate(is_valid):
                        if is_val and valid_solver_outs[idx_batch]:
                            if (
                                pred_is_wrong[idx]
                                and possible_error_due_to_selector[idx_batch]
                            ):
                                is_valid[idx_batch] = False
                            idx += 1

                expression = self.vocabulary.str_to_batch(upd_expression_str)

            if all_low_quality.any():
                for idx in all_low_quality.argwhere().flatten():
                    is_valid[idx] = False
                    self.track_stop_reason["no_valid_leaf"][idx] += 1
        if ~is_valid.any():
            expression = self.get_full_batch(
                expression, is_valid, self.vocabulary.get_special_idx("pad")
            )  # make a batch full of PAD

        expression = self.check_invalid_final_state(expression)
        expression_str = self.vocabulary.batch_to_str(expression)
        logging.info(f"Final expressions {expression_str}")
        expression_solve_vocab = self.vocabulary.str_to_batch(expression_str, x=False)
        return expression_solve_vocab

    def textseg_encoder_fwd(self, expression):
        inputs_str = self.vocabulary.batch_to_str(expression)
        logging.info(f"Inputs:\n{inputs_str}")

        if "listops" in self.vocabulary.tokenizer or "listops" in self.current_task:
            leaf_expr_re = re.compile(r"\[[A-Z]+[\d{1}]+\]")
            atomic_value_re = re.compile(r"[0-9]")
        elif (
            "arithmetic" in self.vocabulary.tokenizer
            or "arithmetic" in self.current_task
        ):
            leaf_expr_re = re.compile(r"^\([+\-*]*[0-9]+[+\-*]*[0-9]+\)$")
            atomic_value_re = re.compile(r"[+-]?[0-9]{1,2}")
        elif "algebra" in self.vocabulary.tokenizer or "algebra" in self.current_task:
            leaf_expr_re = re.compile(
                r"\([+\-]*[0-9]*[abxy*]+[+\-]*[+\-0-9]*[abxy*]+\)"
            )
            atomic_value_re = re.compile(r"[+-][0-9]{0,2}[abxy*]{0,8}")
        elif "char" in self.vocabulary.tokenizer or "logic" in self.current_task:
            leaf_expr_re = re.compile("^\([a-zTF][|&][a-zTF]\)|\(![a-zTF]\)$")
            atomic_value_re = re.compile(r"[a-zTF]")

        output = self.selector_encoder(expression)
        output_tokens = output.argmax(-1)
        # replace non leaf-formulas tokens with pads
        expression[~output_tokens.bool()] = self.vocabulary.get_special_idx("pad")
        modified_input = self.vocabulary.batch_to_str(expression, replace_pad=False)
        leaf_formulas = []
        max_num_candidates = 0
        for idx, modified_formula in enumerate(modified_input):
            modified_formula = re.sub(r"#+", "#", modified_formula)
            candidate_leafs = [
                leaf for leaf in modified_formula.split("#") if leaf != ""
            ]
            upd_candidate_leafs = []
            for leaf in candidate_leafs:
                if len(multi_leafs := leaf_expr_re.findall(leaf)) > 1:
                    upd_candidate_leafs += multi_leafs
                else:
                    upd_candidate_leafs.append(leaf)
            leafs = upd_candidate_leafs
            # leafs = [
            #     leaf
            #     for leaf in candidate_leafs
            #     if leaf_expr_re.fullmatch(leaf) is not None
            #     or atomic_value_re.fullmatch(leaf) is not None
            # ]
            if len(leafs) == 0:
                logging.info(f"No valid candidate leafs for input {inputs_str[idx]}")

            leaf_formulas += [leafs]
            if len(leafs) > max_num_candidates:
                max_num_candidates = len(leafs)

        logging.info(f"Leafs: {leaf_formulas}")

        # right pad
        padded_leafs = []
        for leafs in leaf_formulas:
            if len(leafs) < max_num_candidates:
                missing = max_num_candidates - len(leafs)
                padded_leafs.append(leafs + [None] * missing)
            else:
                padded_leafs.append(leafs)
        return padded_leafs

        # leaf_expr_re = re.compile(r"\([+\-*]*[0-9]+[+\-*]*[0-9]+\)")
        # leaf_expr_re = re.compile(
        #     r"\([+\-]*[0-9]*[abxy*]+[+\-]*[+\-0-9]*[abxy*]+\)"
        # )

    def selector_encoder_fwd(self, expression):
        inputs_str = self.vocabulary.batch_to_str(expression)
        logging.info(inputs_str)

        multi_output_str = [set() for _ in inputs_str]
        for _ in range(self.n_multi):
            output = self.selector_encoder(expression)
            orig_output_str = self.selector_encoder.vocabulary.batch_to_str(
                output.argmax(-1), x=False
            )
            orig_output_str = [o[: len(i)] for o, i in zip(orig_output_str, inputs_str)]

            orig_output_str = [o.replace("][", "]?[") for o in orig_output_str]

            sep_char_re = re.compile(r"\?+")
            split_output_str = [sep_char_re.sub("?", o) for o in orig_output_str]
            split_output_str = [o.split("?") for o in split_output_str]

            # arith_sub_re = re.compile(r'(\([\-+]?\d{1,2}[\-*+][\-+]?\d{1,2}\))|([\-+]?\d{1,2})')
            # algebra_sub_re = re.compile(r'(\([\-+]?\d{0,2}[\-abxy*]*[\-+][\-+]?\d{0,2}[\-abxy*]*\))|([\-+]?\d{0,2}[\-abxy*]*)')

            # output_tokens = output.argmax(-1)
            output_logits = output.max(-1)[0]
            # masked_output_logits = output_logits.clone()
            # idx_sep = self.selector_encoder.vocabulary.get_special_idx('sos', x=False)
            # idx_pad = self.selector_encoder.vocabulary.get_special_idx('pad')
            # masked_output_logits[output_tokens == idx_sep] = 0
            # masked_output_logits[expression == idx_pad] = 0

            # well_formed_output_str = []
            # for input_str, sub_expressions in zip(inputs_str, split_output_str):
            #     well_formed_output_str.append([])
            #     for sub_expression in sub_expressions:
            #         if self._has_well_formed_parentheses(sub_expression) and sub_expression != '' and re.fullmatch(algebra_sub_re, sub_expression) is not None:
            #             if len(sub_expressions) > 1 and re.compile(r'[\-+]?\d{1,2}').fullmatch(sub_expression) is not None:   # ignore fake final states (eg numbers in arithmetic) when they're not the only output given by the selector (ie actual final state)
            #                 continue
            #             well_formed_output_str[-1].append(sub_expression)
            #     if well_formed_output_str[-1] == []:
            #         well_formed_output_str[-1].append('')

            for batch_idx, sub_expressions in enumerate(split_output_str):
                for subexp in sub_expressions:
                    if subexp == "":
                        continue
                    start_pos = orig_output_str[batch_idx].find(subexp)
                    end_pos = start_pos + len(subexp)
                    confidence_score = (
                        output_logits[batch_idx, start_pos:end_pos].mean().item()
                    )
                    multi_output_str[batch_idx].add(
                        (subexp, inputs_str[batch_idx].find(subexp), confidence_score)
                    )

        multi_output_str = [list(multi_out) for multi_out in multi_output_str]

        sub_expression_objs = []
        for input_str, sub_expressions in zip(inputs_str, multi_output_str):
            sub_expressions.sort(key=lambda x: (x[2], x[1]))
            sub_expression, pos, _ = sub_expressions[-1]
            sub_expression_objs.append(
                SelectorOutput(
                    tokenized_text=self.vocabulary.tokenize_sample(sub_expression),
                    confidence_score=1,
                    finder_score=(
                        len(self.vocabulary.tokenize_sample(sub_expression))
                        if sub_expression in input_str
                        else 0
                    ),
                    position_finder=pos,
                )
            )
        return sub_expression_objs


# TODO
# - position with Combiner?
# - solve filter malformed sub-expressions
# - solve listops formatting problem
#
# - train algebra and listops selector
# - retrain solver to 99%
