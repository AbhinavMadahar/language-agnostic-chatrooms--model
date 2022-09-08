import math
import torch

from collections import Iterable, List, OrderedDict, Tuple
from torch import nn


# the following classes are helpers for the Encoder and Decoder classes.
# these should not be used directly.

class EncoderBlock(nn.Module):
    """
    A encoder block consisting of a multi-head attention and a positionwise feedforward layer, both wrapped with residual connections and layer norm.
    """

    def __init__(self, num_hiddens: int, ffn_num_hiddens: int, num_heads: int, dropout: float, use_bias: bool = False) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, valid_lengths: Iterable[int]) -> torch.Tensor:
        raise NotImplementedError


class DecoderBlock(nn.Module):
    """
    A decoder block consisting of a masked multi-head attention, a multi-body attention using the encoder outputs, and a positionwise feedforward layer,
    all wrapped with residual connections and layer norm.
    """

    def __init__(self, num_hiddens: int, ffn_num_hiddens: int, num_heads: int, dropout: float, i: int) -> None:
        raise NotImplementedError
    
    def init_weights(self) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PositionEncoding(nn.Module):
    """
    This layer encodes position information for a transformer model.
    """

    def __init__(self, num_hiddens: int, dropout: float, max_length: int) -> None:
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        self.attention_weights = None

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoding = PositionEncoding(num_hiddens, dropout, max_length)
        self.blocks = nn.Sequential(OrderedDict(
            (f'block {i}', EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias)) for i in range(num_blocks)
        ))
    
    def forward(self, x: torch.Tensor, valid_lengths: Iterable[int]) -> torch.Tensor:
        # don't worry about why we multiply by sqrt(num_hiddens).
        # it's a long story.
        # maybe I'll one day update this comment to explain why.
        x = self.position_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))

        self.attention = []
        # we go pass x through each block
        for block in self.blocks:
            x = block(x, valid_lengths)
            self.attention_weights.append(block.attention.attention.attention_weights)
        
        return x


class Decoder(nn.Module):
    """
    Transformer-based decoder for the machine translation model.
    """

    def __init__(self, vocab_size: int, num_hiddens: int, ffn_num_hiddens: int, num_heads: int, num_blocks: int, dropout: float, max_length: int) -> None:
        super().__init__()

        self.num_hiddens = num_hiddens
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoding = PositionEncoding(num_hiddens, dropout, max_length)

        self.blocks = nn.Sequential(OrderedDict(
            (f'block {i}', DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i)) for i in range(num_blocks)
        ))
        self.dense = nn.LazyLinear(vocab_size)

        self._attention_weights = None

    def init_state(self, encoder_outputs: List[torch.Tensor], encoder_valid_lengths: List[int]) -> None:
        return [encoder_outputs, encoder_valid_lengths, [None] * self.num_blocks]

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.position_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))

        # the first list is the decoder self-attention weights, and the second list is the encoder-decoder attention weights
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