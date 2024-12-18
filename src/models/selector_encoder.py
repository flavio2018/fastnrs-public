import logging
import os

import hydra
import torch
from torch.nn import Embedding, Linear
from models.transformer import Encoder


class SelectorEncoder(torch.nn.Module):

    def __init__(
        self,
        d_model,
        ff_mul,
        num_heads,
        num_layers_enc,
        vocabulary,
        label_pe_enc=False,
        max_range_pe=1000,
        diag_mask_width_below=1,
        diag_mask_width_above=1,
        average_attn_weights=True,
        store_attn_weights=False,
        mha_init_gain=1,
        dropout=0.1,
        use_pe_enc=True,
        device="cuda",
    ):
        super(SelectorEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.device = device
        self.encoder_layers = torch.nn.ModuleList([])
        for _ in range(num_layers_enc):
            self.encoder_layers.append(
                Encoder(
                    d_model,
                    ff_mul,
                    num_heads,
                    dropout=dropout,
                    label_pe=label_pe_enc,
                    max_range_pe=max_range_pe,
                    diag_mask_width_below=diag_mask_width_below,
                    diag_mask_width_above=diag_mask_width_above,
                    average_attn_weights=average_attn_weights,
                    store_attn_weights=store_attn_weights,
                    use_pe=use_pe_enc,
                    device=device,
                )
            )
            self.encoder_layers[-1]._init_mha(mha_init_gain)
        self.d_model = d_model
        self.idx_PAD_x = vocabulary.get_special_idx("pad")
        self.idx_PAD_y = vocabulary.get_special_idx("pad", x=False)
        # self.idx_SOS_y = vocabulary.get_special_idx('sos', x=False)
        # self.idx_EOS_y = vocabulary.get_special_idx('eos', x=False)
        self.len_x_vocab, self.len_y_vocab = len(vocabulary.x_vocab), len(
            vocabulary.y_vocab
        )
        self.x_emb = Embedding(
            num_embeddings=self.len_x_vocab,
            embedding_dim=self.d_model,
            padding_idx=self.idx_PAD_x,
            device=self.device,
        )
        self.final_proj = Linear(self.d_model, self.len_y_vocab, device=self.device)
        for enc in self.encoder_layers:
            enc.set_vocabulary(vocabulary)
        self.store_attn_weights = store_attn_weights

    def load_model_weights(self, ckpt):
        logging.info(f"Loading model weights from checkpoint {ckpt}...")
        torch_ckpt = torch.load(
            os.path.join(hydra.utils.get_original_cwd(), f"../checkpoints/{ckpt}"),
            map_location=self.device,
        )
        self.load_state_dict(torch_ckpt["model"])

    def forward(self, X, Y=None, tf=False, enc_widths=None):
        src_mask = X == self.idx_PAD_x
        X = self.x_emb(X)
        X, src_mask = self._encoder(X, src_mask, widths=enc_widths)
        return self.final_proj(X)

    def _encoder(self, X, src_mask, widths=None):
        for encoder_layer in self.encoder_layers:
            X = encoder_layer(X, src_mask, widths=widths)
        return X, src_mask
