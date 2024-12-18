import logging
import torch
from torch import nn
import torch.nn.functional as F
from models.finder import Finder
from data.vocabulary import EOS, PAD, SEP
from data.generators import ListOpsExpressionGenerator, AlgebraicExpressionGenerator, ArithmeticExpressionGenerator

# from task.simplify_listops import is_final as is_final_listops, is_well_formed as is_well_formed_listops
# from task.simplify_arithmetic import is_final as is_final_arithmetic, is_well_formed as is_well_formed_arithmetic
# from task.simplify_algebra import is_final as is_final_algebra, is_well_formed as is_well_formed_algebra


class NeuralCombiner(nn.Module):
	
	def __init__(self, batch_size):
		super().__init__()
		self.finder = Finder(batch_size)
		
	def get_vocab_info(self, generator):
		self.sep_tok_idx = generator.y_vocab[SEP]
		self.eos_tok_idx = generator.y_vocab[EOS]
		self.pad_tok_idx = generator.y_vocab[PAD]
		self.len_y_vocab = len(generator.y_vocab)
		self.generator = generator
		self.finder.get_vocab_info(generator)
		
	def _to_1hot(self, batch):
		return F.one_hot(batch.to(torch.int64), num_classes=self.len_y_vocab).type(torch.float)
	
	def _is_final(self, X):
		if isinstance(self.generator, AlgebraicExpressionGenerator):
			return is_final_algebra(self.generator, X)

		elif isinstance(self.generator, ArithmeticExpressionGenerator):
			return is_final_arithmetic(self.generator, X)

		elif isinstance(self.generator, ListOpsGenerator):
			return is_final_listops(self.generator, X)

	def _is_well_formed(self, output):
		if isinstance(self.generator, ListOpsGenerator):
			return is_well_formed_listops(self.generator, output)
		
		elif isinstance(self.generator, AlgebraicExpressionGenerator):
			return is_well_formed_algebra(self.generator, output)

		elif isinstance(self.generator, ArithmeticExpressionGenerator):
			return is_well_formed_arithmetic(self.generator, output)
	
	def forward(self, X, solver_out):
		final = self._is_final(X)

		well_formed = self._is_well_formed(X)
		logging.info("Well-formed")
		logging.info(well_formed)

		splittable = self._is_splittable(solver_out)
		logging.info("Splittable")
		logging.info(splittable)
		
		splittable_and_well_formed = torch.logical_and(splittable, well_formed)
		valid = torch.logical_and(splittable_and_well_formed, ~final)

		if valid.any():  # if there is any valid output
			valid_X = X[valid]
			valid_solver_out = solver_out[valid]
			expression, result, expression_length, result_length = self._split(valid_solver_out)
			
			# prevent padded values from matching in the filter
			expression_mask = (expression == self.pad_tok_idx)
			expression_mask_3d = expression_mask.unsqueeze(-1).tile(1, 1, self.len_y_vocab)
			expression_1hot = self._to_1hot(expression)
			expression_1hot[expression_mask_3d] = 0
			
			start_signal = self.finder(valid_X, expression_1hot)
			output_tokens = self._pool(valid_X, start_signal, expression_length, result_length, result)

			if (~splittable).any() or (~well_formed).any():
				output = torch.full((X.size(0), X.size(1)), self.pad_tok_idx, device=X.device)
				output[valid, :output_tokens.size(1)] = output_tokens

				if final.any():
					output[final] = X[final]
				
				if (~splittable).any():
					output[~splittable] = X[~splittable]

				if (~well_formed).any():
					output[~well_formed] = X[~well_formed]

			elif splittable.all() and well_formed.all():
				output = torch.full((X.size(0), output_tokens.size(1)), self.pad_tok_idx, device=X.device)
				output[valid] = output_tokens

		else:

			if (~splittable).any() or (~well_formed).any():
				output = torch.full((X.size(0), X.size(1)), self.pad_tok_idx, device=X.device)
				
				if final.any():
					output[final] = X[final]
				
				if (~splittable).any():
					output[~splittable] = X[~splittable]

				if (~well_formed).any():
					output[~well_formed] = X[~well_formed]
			
			else:
				raise RuntimeError("No valid solver output but all inputs are well-formed and all solver outputs are splittable.")

		return self._to_1hot(output)


	def _is_splittable(self, X):
		mask_eos_tok = (X == self.eos_tok_idx)
		exists_one_eos_tok_per_seq = (mask_eos_tok.sum(1) >= 1)

		mask_first_eos_tok = ((X == self.eos_tok_idx).cumsum(1).cumsum(1) == 1)
		valid_mask_first_eos = mask_first_eos_tok[exists_one_eos_tok_per_seq]
		positions_batch = torch.ones_like(X).cumsum(1) - 1
		valid_positions_batch = positions_batch[exists_one_eos_tok_per_seq]
		mask_before_first_eos = valid_positions_batch < valid_positions_batch[valid_mask_first_eos].unsqueeze(1)
		mask_sep_tok = (X == self.sep_tok_idx)
		valid_mask_sep_tok = mask_sep_tok[exists_one_eos_tok_per_seq]
		valid_mask_sep_tok[~mask_before_first_eos] = False
		exactly_one_sep_before_first_eos = torch.zeros_like(exists_one_eos_tok_per_seq).bool()
		exactly_one_sep_before_first_eos[exists_one_eos_tok_per_seq] = valid_mask_sep_tok.sum(-1) == 1
		
		return torch.bitwise_and(exists_one_eos_tok_per_seq, exactly_one_sep_before_first_eos)
		

	def _split(self, X):
		"""Splits input batch in a batch of expressions and one of results.
		
		The method operates in the following way: first, it creates a new 
		batch of expressions and then one of results.
		
		The first one is created starting from a batch full of padding value,
		which is overwritten with true expressions only on valid positions.
		The second one is built gathering the result and padding tokens
		from the input into a new tensor."""
		
		# build mask of positions of first occurrence of SEP.
		mask_sep_tok = ((X == self.sep_tok_idx).cumsum(1).cumsum(1) == 1)

		# build batch of expressions
		n_cols_expr_batch = torch.argwhere(mask_sep_tok)[:, 1].max()
		positions_expr_batch = torch.tensor(range(n_cols_expr_batch)).tile((X.size(0), 1)).to(X.device)
		expressions_lengths = torch.argwhere(mask_sep_tok)[:, 1]
		end_expr_pos = expressions_lengths.unsqueeze(1)
		expr_mask = positions_expr_batch < end_expr_pos
		fill_pad_expr_batch = torch.zeros((X.size(0), n_cols_expr_batch)).fill_(self.pad_tok_idx).to(X.device)
		expr_batch = torch.where(expr_mask, X[:, :n_cols_expr_batch], fill_pad_expr_batch)

		# build batch of results
		positions_X = torch.tensor(range(X.size(1))).tile((X.size(0),1)).to(X.device)
		end_expr_pos = torch.argwhere(mask_sep_tok)[:, 1].unsqueeze(1)
		# build mask of positions of first occurrence of EOS.
		mask_eos_tok = ((X == self.eos_tok_idx).cumsum(1).cumsum(1) == 1)
		# there should be one and only one value per row
		assert mask_eos_tok.sum() == mask_eos_tok.size(0), f"Num EOS tokens {mask_eos_tok.sum()} != num rows {mask_eos_tok.size(0)}"
		eos_pos = torch.argwhere(mask_eos_tok)[:, 1].unsqueeze(1)
		max_result_length = (eos_pos - end_expr_pos - 1).max()  # excluding EOS
		fill_pad_res_batch = torch.zeros((X.size(0), max_result_length)).fill_(self.pad_tok_idx).to(X)
		res_mask_solver_out = torch.bitwise_and(end_expr_pos < positions_X, positions_X < eos_pos)
		results_lengths = res_mask_solver_out.sum(1)
		res_batch_positions = torch.ones_like(fill_pad_res_batch).cumsum(1) - 1
		res_mask_res_batch = res_batch_positions < eos_pos - end_expr_pos - 1
		fill_pad_res_batch[res_mask_res_batch] = X[res_mask_solver_out]
		
		return expr_batch, fill_pad_res_batch, expressions_lengths, results_lengths

	def _pool(self, old_input, start_signal, expressions_lengths, results_lengths, res_batch):
		"""Pools sub-expression in original input with its result.
		
		The method operates in the following way: first, it builds
		a batch of new input filled with padding values. Then, it
		progressively substitutes those with parts of the original
		input expressions and with results that replace the 
		solved sub-expressions. The replacement of padding values
		makes heavy use of masking operations.
		"""
		old_input_nopad_mask = (old_input != self.pad_tok_idx)
		old_input_lengths = old_input_nopad_mask.sum(1)

		# fixes hallucinations where expressions are longer than the original input
		expressions_lengths = torch.min(old_input_lengths, expressions_lengths)
		
		# fix error when start signal is beyond end of old input and leaves no space for substitution
		where_signal_beyond_valid_end = (old_input_lengths.unsqueeze(1) - start_signal) < results_lengths.unsqueeze(1)
		while where_signal_beyond_valid_end.any():
			start_signal[where_signal_beyond_valid_end] = start_signal[where_signal_beyond_valid_end] - 1
			where_signal_beyond_valid_end = (old_input_lengths.unsqueeze(1) - start_signal) < results_lengths.unsqueeze(1)
		start_signal[start_signal < 0] = 0

		# fix error when the substitution would go beyond end of old input
		where_substitution_beyond_valid_end = (start_signal + results_lengths.unsqueeze(1)) > old_input_lengths.unsqueeze(1)
		if where_substitution_beyond_valid_end.any():
			start_signal[where_substitution_beyond_valid_end] = start_signal[where_substitution_beyond_valid_end] - 1
			where_substitution_beyond_valid_end = (start_signal + results_lengths.unsqueeze(1)) > old_input_lengths.unsqueeze(1)
		start_signal[start_signal < 0] = 0

		new_input_lengths = old_input_lengths - expressions_lengths + results_lengths
		new_input = (torch.full((old_input.size(0), new_input_lengths.max()), self.pad_tok_idx).to(old_input))
		new_input_positions = torch.ones_like(new_input).cumsum(1) - 1
		old_input_positions = torch.ones_like(old_input).cumsum(1) - 1

		# compute in advance
		end_expr_old_input = start_signal + expressions_lengths.unsqueeze(1) - 1
		end_expr_new_input = (end_expr_old_input
							  - expressions_lengths.unsqueeze(1)
							  + results_lengths.unsqueeze(1) + 1)
		
		if (end_expr_new_input > new_input.size(1)).any():  # fixes halluciations where expression result gets inserted beyond new input length
			where_greater = end_expr_new_input > new_input.size(1)
			diff = new_input.size(1) - end_expr_new_input[where_greater]
			end_expr_new_input[where_greater] -= diff
			start_signal[where_greater] -= diff
	
		# insert start expression
		mask_start_old_input = (old_input_positions <= start_signal)
		
		if mask_start_old_input.size(1) > new_input.size(1):
			mask_start_old_input_for_new_input = mask_start_old_input[:, :new_input.size(1)]
		
		elif mask_start_old_input.size(1) < new_input.size(1):
			diff = abs(mask_start_old_input.size(1) - new_input.size(1))
			mask_start_old_input_for_new_input = torch.cat([mask_start_old_input, torch.zeros(new_input.size(0), diff).bool().to(mask_start_old_input)], dim=1) 
		
		else:
			mask_start_old_input_for_new_input = mask_start_old_input
		new_input[mask_start_old_input_for_new_input] = old_input[mask_start_old_input]

		# insert result
		new_input_res_mask = torch.bitwise_and(new_input_positions >= start_signal,
											   new_input_positions < end_expr_new_input)

		new_input[new_input_res_mask] = res_batch[res_batch != self.pad_tok_idx].squeeze()

		# insert end expression 
		mask_end_expr_old_input = torch.bitwise_and(old_input_positions > end_expr_old_input, old_input_nopad_mask)
		mask_end_expr_new_input = torch.bitwise_and(new_input_positions >= end_expr_new_input,
													new_input_positions < new_input_lengths.unsqueeze(1))
		new_input[mask_end_expr_new_input] = old_input[mask_end_expr_old_input]

		return new_input
