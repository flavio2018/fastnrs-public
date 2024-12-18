import hydra
import os
import logging
import torch
from models.transformer import Encoder
from torch.nn import Embedding, Linear


class WidthRegressor(torch.nn.Module):

	def __init__(self, n_layers, vocabulary, d_model, ff_mul, num_heads, dropout=0.1, use_pe=False, label_pe=True, max_range_pe=1000, device='cpu'):
		super(WidthRegressor, self).__init__()
		self.vocabulary = vocabulary
		self.x_emb = Embedding(num_embeddings=len(vocabulary.x_vocab), embedding_dim=d_model, padding_idx=self.vocabulary.get_special_idx('pad'), device=device)
		self.encoder_layers = torch.nn.ModuleList([])
		for _ in range(n_layers):
			self.encoder_layers.append(
				Encoder(d_model, ff_mul, num_heads, dropout=dropout, label_pe=label_pe, max_range_pe=max_range_pe, use_pe=use_pe, device=device))
		self.regr_proj = Linear(d_model, 1, device=device)
		self.device = device
	
	def forward(self, X):
		src_mask = (X == self.vocabulary.get_special_idx('pad'))
		X = self.x_emb(X)

		for encoder_layer in self.encoder_layers:
			X = encoder_layer(X, src_mask)	# (bs, seq_len, d_model)
		X = X.mean(dim=1)  # (bs, d_model)
		return self.regr_proj(X).squeeze()	# (bs, 1)
	
	def load_model_weights(self, ckpt):
		logging.info(f'Loading model weights from checkpoint {ckpt}...')
		torch_ckpt = torch.load(os.path.join(hydra.utils.get_original_cwd(), f'../checkpoints/{ckpt}'), map_location=self.device)
		self.load_state_dict(torch_ckpt['model'])
