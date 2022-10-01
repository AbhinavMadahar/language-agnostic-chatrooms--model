import math
import torch

from typing import Iterable, List, OrderedDict, Tuple
from torch import nn


# the following classes are helpers for the Encoder and Decoder classes.
# these should not be used directly.


def masked_softmax(X: torch.Tensor, valid_lengths: List[int]) -> torch.Tensor:
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lengths is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lengths.dim() == 1:
            valid_lengths = torch.repeat_interleave(valid_lengths, shape[1])
        else:
            valid_lengths = valid_lengths.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lengths, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout: float, num_heads: int = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None, window_mask=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = scores.reshape((n // (num_windows * self.num_heads),
                                     num_windows,
                                     self.num_heads,
                                     num_queries,
                                     num_kv_pairs)) \
                    + window_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.reshape((n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, num_hiddens: int, num_heads: int, dropout: float, bias: bool = False) \
        -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens, window_mask=None):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens,
                                window_mask)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input x:
        #   (batch_size, no. of queries or key-value pairs, # num_hiddens).
        # Shape of output x:
        #   (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        # Shape of output x:
        #   (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        x = x.permute(0, 2, 1, 3)
        # Shape of output:
        #   (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def transpose_output(self, x):
        """Reverse the operation of transpose_qkv."""
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens: int, ffn_num_outputs: int) -> None:
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(y) + x)


class DecoderBlock(nn.Module):
    """
    A decoder block consisting of a masked multi-head attention, a multi-body attention using the
    encoder outputs, and a positionwise feedforward layer, all wrapped with residual connections
    and layer norm.
    """

    def __init__(self,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 dropout: float,
                 i: int) -> None:
        super().__init__()

        self.i = i

        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> torch.Tensor:
        encoder_outputs, encoder_valid_lengths = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step

        key_values = x if state[2][self.i] is None else torch.cat((state[2][self.i], x), dim=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = x.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        x2 = self.attention1(x, key_values, key_values, dec_valid_lens)
        y = self.addnorm1(x, x2)

        # Encoder-decoder attention. Shape of enc_outputs: (batch_size, num_steps, num_hiddens)
        y2 = self.attention2(y, encoder_outputs, encoder_outputs, encoder_valid_lengths)
        z = self.addnorm2(y, y2)
        x3 = self.addnorm3(z, self.ffn(z))

        return x3, state


class EncoderBlock(nn.Module):
    """
    A encoder block consisting of a multi-head attention and a positionwise feedforward layer,
    both wrapped with residual connections and layer norm.
    """

    def __init__(self,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 dropout: float,
                 use_bias: bool = False) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, x: torch.Tensor, valid_lengths: Iterable[int]) -> torch.Tensor:
        y = self.addnorm1(x, self.attention(x, x, x, valid_lengths))
        z = self.addnorm2(y, self.ffn(y))
        return z


class PositionEncoding(nn.Module):
    """
    This layer encodes position information for a transformer model.
    """

    SUPER_LONG_NUMBER = 10000

    def __init__(self, num_hiddens: int, dropout: float, max_length: int) -> None:
        if max_length > self.SUPER_LONG_NUMBER:
            raise ValueError(
                f'The maximum supported max_length is {self.SUPER_LONG_NUMBER}, but' + \
                f'max_length={max_length} was passed as an argument. '
            )
        if num_hiddens % 2 != 0:
            raise ValueError(f"num_hiddens must be even, but {num_hiddens} supplied.")

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_length, num_hiddens))
        x = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(
                self.SUPER_LONG_NUMBER,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


# we implement the main Encoder and Decoder classes now.
# these are the only classes that should be used outside this module.

class Encoder(nn.Module):
    """
    Transformer-based encoder for the machine translation model.
    """

    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_blocks: int,
                 dropout: float,
                 max_length: int,
                 use_bias: bool = False
        ) -> None:
        super().__init__()

        self.num_hiddens = num_hiddens
        self.attention_weights: List = None

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoding = PositionEncoding(num_hiddens, dropout, max_length)
        self.blocks = nn.Sequential(OrderedDict(
            (f'block {i}', EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
            for i in range(num_blocks)
        ))

    def forward(self, x: torch.Tensor, valid_lengths: List[int]) -> torch.Tensor:
        """
        Send sequences through the transformer.

        Args:
            x (torch.Tensor): A tensor representing the sequences.
                              It should have shape (number of sequences, max length of sequence).
            valid_lengths (List[int]): _description_

        Returns:
            torch.Tensor: The final output of the all the blocks of the transformer.
        """
        # don't worry about why we multiply by sqrt(num_hiddens).
        # it's a long story.
        # maybe I'll one day update this comment to explain why.
        x = self.position_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))

        self.attention_weights = []
        # we go pass x through each block
        for block in self.blocks:
            x = block(x, valid_lengths)
            self.attention_weights.append(block.attention.attention.attention_weights)

        return x


class Decoder(nn.Module):
    """
    Transformer-based decoder for the machine translation model.
    """

    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_blocks: int,
                 dropout: float,
                 max_length: int) -> None:
        super().__init__()

        self.num_hiddens = num_hiddens
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoding = PositionEncoding(num_hiddens, dropout, max_length)

        self.blocks = nn.Sequential(OrderedDict(
            (f'block {i}', DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
            for i in range(num_blocks)
        ))
        self.dense = nn.LazyLinear(vocab_size)

        self._attention_weights = None

    def init_state(self, encoder_outputs: List[torch.Tensor], encoder_valid_lengths: List[int]) \
        -> None:
        return [encoder_outputs, encoder_valid_lengths, [None] * self.num_blocks]

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.position_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))

        # the first list is the decoder self-attention weights,
        # and the second list is the encoder-decoder attention weights
        self._attention_weights = [[], []]
        for i, block in enumerate(self.blocks):
            x, state = block(x, state)
            self._attention_weights[0].append(block.attention1.attention.attention_weights)
            self._attention_weights[1].append(block.attention2.attention.attention_weights)

        x = self.dense(x)

        return x, state

    @property
    def attention_weights(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._attention_weights
