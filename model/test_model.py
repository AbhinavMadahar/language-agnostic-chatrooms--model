import torch

from model import Encoder
from model import PositionEncoding


def test_position_encoding():
    num_hiddens, dropout, max_length = 12, 0.5, 20
    position_encoding = PositionEncoding(num_hiddens, dropout, max_length)


def test_encoder():
    num_hiddens = 12

    encoder = Encoder(vocab_size=91,
                      num_hiddens=num_hiddens,
                      ffn_num_hiddens=17,
                      num_heads=3,
                      num_blocks=3,
                      dropout=0.5,
                      max_length=20,
                      use_bias=True)

    input_sentences = torch.Tensor([
        [1, 3, 2, 0, 34, 0],
        [1, 3, 2, 0, 0, 0],
        [1, 3, 2, 2, 0, 0],
    ]).to(torch.int)

    valid_lengths = torch.Tensor([5, 3, 4])
    x = encoder(input_sentences, valid_lengths)

    assert x.shape == (*input_sentences.shape, num_hiddens)
