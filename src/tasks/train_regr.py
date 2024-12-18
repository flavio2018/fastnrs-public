import hydra
import os
import torch
from tasks.base import BaseTask
from tasks.mixins import EvalTaskMixin, VisualizeTaskMixin


class TrainRegrTask(BaseTask, VisualizeTaskMixin, EvalTaskMixin):

	def __init__(self, model, datasets, cfg):
		super(TrainRegrTask, self).__init__(model, datasets, cfg)
		self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.task.lr)
		self.criterion = torch.nn.MSELoss(reduction="mean")
		self.reset_metrics_dict()
		
	def run(self):
		super().run()
		self.train()

	def train(self):
		for it in range(self.cfg.task.max_iter):
			self.train_step()

			if it % self.FREQ_WANDB_LOG == 0:
				self.valid_step_regr()
				self.reset_metrics_dict()
				self.serialize(it)

	def train_step(self):
		self.model.train()
		self.opt.zero_grad()
		X, Y = next(iter(self.dataloaders['train']))
		output = self.model(X)
		loss = self.criterion(output, Y)
		loss.backward()
		self.opt.step()
		self.valid_step_metrics['metrics/train/mse_loss'] = loss.item()

	def serialize(self, it):
		torch.save({
					'update': it,
					'model': self.model.state_dict(),
					'model_cfg': self.cfg.model,
					'opt': self.opt.state_dict(),
				}, os.path.join(hydra.utils.get_original_cwd(), f"../checkpoints/{self.cfg.start_timestamp}_{self.cfg.name}.pth"))
