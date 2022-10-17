import torch
import random

from scipy.stats import linregress
from torch import nn
from typing import Generator, Tuple

from data import read
from model import Encoder, Decoder, EncoderDecoderModel
from train import train_many_to_many, train_many_to_one, train_single_epoch
from vocab import Vocabulary


def test_train_single_epoch() -> None:
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

    train_single_epoch(model,
                       data(),
                       optim,
                       criterion,
                       validation_split=0.2,
                       batch_size=128,
                       num_batches=200,
                       vocab=vocab,
                       bidirectional=True)


def test_train_many_to_many() -> None:
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

    each_epoochs_train_losses, validation_losses = train_many_to_many(model,
                                                                      data(),
                                                                      optim,
                                                                      criterion,
                                                                      validation_split=0.2,
                                                                      batch_size=128,
                                                                      vocab=vocab,)

    # make sure that the train and losses go down over time
    train_losses = sum(each_epoochs_train_losses, [])
    train_regression = linregress(list(enumerate(train_losses)))
    assert train_regression.slope < 0, train_losses

    validation_regression = linregress(list(enumerate(validation_losses)))
    assert validation_regression.slope < 0, validation_losses


def test_train_many_to_one() -> None:
    """
    Trains the model during the second phase.

    Arguments:
        model: The model to train.
        data: A stream of input and target sequences and their lengths.
        optim: The optimizer to use.
        criterion: A loss function.
        learning_rate: The learning rate to use during optimization.
        validation_split: How much of the data to use for validation as a proportion from 0 to 1.
                          This cannot be equal to 1.
        batch_size: The batch size to use during training.
        num_batches: How many batches to go through before validation.
        vocab: The vocabulary to use during training.

    Returns:
        A tuple of two lists. The first list contains the training losses grouped by epoch and the
        second list contains the validation losses.

    Raises:
        ValueError: If validation_split is not in [0, 1).
    """
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
        target_language = 'yo'

        language_pairs_which_involve_target_language = set(
            (input_, target) for input_, target in multilingual_data_stream.keys()
            if input_ == target_language or target == target_language)

        while True:
            try:
                language_pair = random.sample(language_pairs_which_involve_target_language, k=1)[0]
            except AttributeError:
                # this error means that language_pairs_which_involve_target_language is empty,
                # which means that we have gone through all the data we have. at this point, we
                # should just stop.
                break
            try:
                pair = next(multilingual_data_stream[language_pair])
                yield pair
            except StopIteration:
                del multilingual_data_stream[language_pair]
                language_pairs_which_involve_target_language.remove(language_pair)

    optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    each_epochs_train_losses, validation_losses = train_many_to_one(model,
                                                                    data(),
                                                                    optim,
                                                                    criterion,
                                                                    validation_split=0.2,
                                                                    batch_size=128,
                                                                    vocab=vocab,)

    # make sure that the train and losses go down over time
    train_losses = sum(each_epochs_train_losses, [])
    train_regression = linregress(list(enumerate(train_losses)))
    assert train_regression.slope < 0, train_losses

    validation_regression = linregress(list(enumerate(validation_losses)))
    assert validation_regression.slope < 0, validation_losses
