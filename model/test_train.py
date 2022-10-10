import math
import torch
import random

from torch import nn
from typing import Generator, Tuple

from data import read
from model import Encoder, Decoder, EncoderDecoderModel
from train import rouge_n, train_many_to_many_single_epoch
from vocab import Vocabulary

def test_train_many_to_many_single_epoch() -> None:
    vocab = Vocabulary('data/vocabulary.vocab')

    num_hiddens = 12
    max_length = 20

    encoder = Encoder(vocab_size=vocab.size,
                      num_hiddens=num_hiddens,
                      ffn_num_hiddens=17,
                      num_heads=3,
                      num_blocks=3,
                      dropout=0.5,
                      max_length=max_length,
                      use_bias=True)

    decoder = Decoder(vocab_size=vocab.size,
                      num_hiddens=num_hiddens,
                      ffn_num_hiddens=11,
                      num_heads=4,
                      num_blocks=3,
                      dropout=0.5,
                      max_length=max_length)

    model = EncoderDecoderModel(encoder, decoder)

    def data() -> Generator[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]], None, None]:
        multilingual_data_stream = read(['ar', 'yo'], max_length, vocab)
        while multilingual_data_stream:
            language_pair = random.sample(multilingual_data_stream.keys(), k=1)[0]
            try:
                pair = next(multilingual_data_stream[language_pair])
                yield pair
            except StopIteration:
                del multilingual_data_stream[language_pair]

    optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    train_many_to_many_single_epoch(model,
                                    data(),
                                    optim,
                                    criterion,
                                    validation_split=0.2,
                                    batch_size=128,
                                    num_batches=200,
                                    vocab=vocab,)

