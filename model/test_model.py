import torch

from model import Encoder, Decoder, PositionEncoding, EncoderBlock, DecoderBlock


def test_position_encoding():
    num_hiddens, dropout, max_length = 12, 0.5, 20
    PositionEncoding(num_hiddens, dropout, max_length)


def test_encoder_block() -> None:
    encoder_block = EncoderBlock(num_hiddens=24,
                                 ffn_num_hiddens=48,
                                 num_heads=8,
                                 dropout=0.5,
                                 use_bias=False)
    encoder_block.eval()

    x = torch.ones((2, 100, 24))
    valid_lengths = torch.tensor([3, 2])
    y = encoder_block(x, valid_lengths)
    assert y.shape == x.shape


def test_decoder_block() -> None:
    encoder_block = EncoderBlock(24, 48, 8, 0.5)
    encoder_block.eval()

    decoder_block = DecoderBlock(num_hiddens=24,
                                 ffn_num_hiddens=48,
                                 num_heads=8,
                                 dropout=0.5,
                                 i=0)
    x = torch.rand((2, 100, 24))
    valid_lengths = torch.tensor([3, 2])
    state = [x, valid_lengths, [None]]
    y = decoder_block(x, state)[0]

    assert y.shape == x.shape


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
    z = encoder(input_sentences, valid_lengths)

    assert z.shape == (*input_sentences.shape, num_hiddens)


def test_decoder() -> None:
    num_blocks = 4
    num_hiddens = 12

    decoder = Decoder(vocab_size=100,
                      num_hiddens=num_hiddens,
                      ffn_num_hiddens=10,
                      num_heads=4,
                      num_blocks=num_blocks,
                      dropout=0.5,
                      max_length=10)

    encoder_outputs = torch.rand((3, 6, num_hiddens))
    valid_lengths = torch.Tensor([5, 3, 4])
    state = decoder.init_state(encoder_outputs, valid_lengths)

    input_sentences = torch.Tensor([
        [1, 3, 2, 0, 34, 0],
        [1, 3, 2, 0, 0, 0],
        [1, 3, 2, 2, 0, 0],
    ]).to(torch.int)

    decoder(input_sentences, state)
