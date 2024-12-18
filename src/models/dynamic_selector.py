import torch


class DynamicSelector(torch.nn.Module):

	def __init__(self, width_regressor, selector):
		super(DynamicSelector, self).__init__()
		self.width_regressor = width_regressor
		self.selector = selector
		for param in self.width_regressor.parameters():
			param.requires_grad = False

	def forward(self, X, Y=None, tf=False):
		enc_widths = self.width_regressor(X)
		enc_widths = torch.round(enc_widths)
		return self.selector(X, Y, tf=tf, enc_widths=enc_widths)
	
	def state_dict(self):
		return self.selector.state_dict()

	def parameters(self):
		return self.selector.parameters()