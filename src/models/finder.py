import logging
import torch
from torch import nn
from data.vocabulary import SEP, PAD


class Finder(nn.Module):

    def __init__(self, batch_size):
        super().__init__()
        self._define_conv2d(batch_size)
        
    def _define_conv2d(self, batch_size):
        self.conv2d = nn.Conv2d(
            in_channels=batch_size,
            out_channels=batch_size,
            kernel_size=(1, 100),  # this is just a placeholder
            padding='valid',
            groups=batch_size)

    def get_vocab_info(self, vocabulary):
        self.sep_tok_idx = vocabulary.y_vocab[SEP]
        self.pad_tok_idx = vocabulary.y_vocab[PAD]
        self.vocabulary = vocabulary

    def _get_output_conv2d(self, X, expressions):
        self._define_conv2d(X.size(0))
        expressions = self._fix_expression_if_longer_than_X(expressions, X)
        self._prepare_custom_filter(expressions)

        filter_length = self.conv2d.weight.size(2)
        num_filler_paddings = filter_length
        pad_1hot = torch.zeros((1, 1, X.size(2)), device=X.device)
        pad_1hot[:, :, self.pad_tok_idx] = 1
        filler_paddings = pad_1hot.tile((X.size(0), num_filler_paddings, 1))
        X_filler_paddings = torch.cat((X, filler_paddings), 1)

        return self.conv2d(X_filler_paddings.unsqueeze(dim=0))

    def forward(self, X, expressions):
        conv2d_out = self._get_output_conv2d(X, expressions)

        # first match
        return conv2d_out.argmax(dim=2).squeeze(0)

        # last match
        # squeezed_conv2d_out = conv2d_out.squeeze()

        # if (X.size(0) == 1) and (squeezed_conv2d_out.dim() == 0):  # handles case of exact filter-input match with one element
        #     return torch.zeros_like(squeezed_conv2d_out).unsqueeze(0).unsqueeze(1)

        # elif (X.size(0) == 1) and (squeezed_conv2d_out.dim() > 0):
        #     squeezed_conv2d_out = squeezed_conv2d_out.unsqueeze(0)

        # elif (len(squeezed_conv2d_out.size()) == 1):
        #     squeezed_conv2d_out = squeezed_conv2d_out.unsqueeze(1)
        
        # element_equals_max = squeezed_conv2d_out == squeezed_conv2d_out.max(dim=1).values.unsqueeze(dim=1)
        # flipped_element_equals_max = torch.flip(element_equals_max, dims=[1])
        # reversed_position_last_max = torch.argwhere(flipped_element_equals_max.cumsum(1).cumsum(1) == 1)
        # position_last_max = element_equals_max.size(1) - reversed_position_last_max[:, 1] - 1
        # return position_last_max.unsqueeze(1)

    def get_expressions_match(self, X, expressions):
        conv2d_out = self._get_output_conv2d(X, expressions)
        # first match
        values, indices = conv2d_out.max(dim=2)
        return values.squeeze()
        
        # last match
        # squeezed_conv2d_out = conv2d_out.squeeze()

        # if (X.size(0) == 1) and (squeezed_conv2d_out.dim() == 0):  # handles case of exact filter-input match with one element
        #     return torch.zeros_like(squeezed_conv2d_out).unsqueeze(0).unsqueeze(1)

        # elif (X.size(0) == 1) and (squeezed_conv2d_out.dim() > 0):
        #     squeezed_conv2d_out = squeezed_conv2d_out.unsqueeze(0)

        # elif (len(squeezed_conv2d_out.size()) == 1):
        #     squeezed_conv2d_out = squeezed_conv2d_out.unsqueeze(1)

        # element_equals_max = squeezed_conv2d_out == squeezed_conv2d_out.max(dim=1).values.unsqueeze(dim=1)
        # flipped_element_equals_max = torch.flip(element_equals_max, dims=[1])
        # reversed_position_last_max = torch.argwhere(flipped_element_equals_max.cumsum(1).cumsum(1) == 1)
        # position_last_max = element_equals_max.size(1) - reversed_position_last_max[:, 1] - 1
        # return squeezed_conv2d_out[torch.arange(squeezed_conv2d_out.size(0)), position_last_max]

    def _fix_expression_if_longer_than_X(self, expressions, X):
        """If the expressions are longer than the input, they
        are cut to the maximum input length.
        This prevents the conv2d op from failing when expressions 
        (i.e. the filter of the op) are longer than the input, which is 
        not permitted.
        """
        if expressions.size(1) > X.size(1):
            expressions = expressions[:, :X.size(1), :]
        return expressions

    def _prepare_custom_filter(self, expressions):
        # cut expressions at sep position to convolve over the whole input seq
        mask_sep_tok = ((expressions.argmax(-1) == self.sep_tok_idx).cumsum(1).cumsum(1) == 1)
        sep_tok_pos = torch.argwhere(mask_sep_tok)[:, 1].unsqueeze(1)
        if mask_sep_tok.any(dim=-1).all():  # if there's at least one sep char for each seq
            expressions = expressions.argmax(-1)
            column_to_slice = torch.max(sep_tok_pos)
            sliced_expressions = expressions[:, :column_to_slice]
            batch_positions = torch.ones_like(mask_sep_tok).cumsum(-1) - 1
            positions_after_sep = batch_positions >= sep_tok_pos
            positions_after_sep = positions_after_sep[:, :column_to_slice]
            sliced_expressions[positions_after_sep] = self.vocabulary.y_vocab['.']
            sliced_expressions = nn.functional.one_hot(sliced_expressions, num_classes=len(self.vocabulary.y_vocab)).type(torch.float)
        else:
            sliced_expressions = expressions
        
        # logging.info(self.vocabulary.batch_to_str(sliced_expressions, x=False))
        custom_filter = sliced_expressions.unsqueeze(1)
        self.conv2d.weight = nn.Parameter(custom_filter)
        self.conv2d.bias = nn.Parameter(torch.tensor([0.0]*len(sliced_expressions), device=sliced_expressions.device))
    