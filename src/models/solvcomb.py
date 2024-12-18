import logging
import torch
from torch import nn
from models.transformer import Transformer
from models.combiner import NeuralCombiner


class SolverCombiner(nn.Module):

	def __init__(self, solver, combiner):
		super().__init__()
		self.solver = solver
		self.combiner = combiner

	def get_vocab_info(self, vocabulary):
		self.solver.get_vocab_info(vocabulary)
		self.combiner.get_vocab_info(vocabulary)
		self.vocabulary = vocabulary

	def forward(self, X, Y=None, tf=False):
		solver_output = self.solver(X, Y=Y, tf=tf)
		logging.info("Solver out")
		logging.info(self.vocabulary.batch_to_str(solver_output, x=False))
		combiner_output = self.combiner(X, solver_output)
		return solver_output, combiner_output

	def load_state_dict(self, state_dict):
		self.solver.load_state_dict(state_dict)

	def state_dict(self):
		return self.solver.state_dict()
