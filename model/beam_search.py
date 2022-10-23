import argparse
import random
import math
import numpy
import torch

from functools import partial
from typing import Callable, List, NamedTuple

from data import read
from model import Encoder, Decoder
from train import train
from vocab import Vocabulary


class Configuration(NamedTuple):
    batch_size: int

    encoder_num_hiddens: int
    encoder_ffn_num_hiddens: int
    encoder_num_heads: int
    encoder_num_blocks: int
    encoder_dropout: float
    encoder_use_bias: bool

    decoder_num_hiddens: int
    decoder_ffn_num_hiddens: int
    decoder_num_heads: int
    decoder_num_blocks: int
    decoder_dropout: float

    first_phase_learning_rate: float
    second_phase_learning_rate: float


def beam_search_pass(beam: List[Configuration],
                     descend: Callable[[Configuration], Configuration],
                     evaluate: Callable[[Vocabulary, List[str], torch.device, int, float, Configuration],
                                        float],
                     vocab: Vocabulary,
                     languages: List[str],
                     beam_size: int,
                     expansion_factor: int,
                     device: torch.device,
                     max_length: int,
                     validation_split: float,
    ) -> List[Configuration]:
    """
    Given a list of configurations, performs a beam search pass.
    """

    children = [descend(configuration) for _ in range(expansion_factor) for configuration in beam]
    children: list[Configuration] = sorted(children, key=partial(evaluate, vocab, languages, device, max_length, validation_split,))
    children = children[:beam_size]

    return children


def descend(parent: Configuration) -> Configuration:
    """
    Given a configuration, find a similar configuration.
    """
    batch_size    = int(parent.batch_size * (2 * random.random()))

    encoder_num_hiddens     = max(int(parent.encoder_num_hiddens * (2 * random.random())), 2)
    if encoder_num_hiddens % 2 != 0:
        encoder_num_hiddens += 1
    encoder_ffn_num_hiddens = max(int(parent.encoder_ffn_num_hiddens * (2 * random.random())), 1)
    encoder_num_heads       = max(int(parent.encoder_num_heads * (2 * random.random())), 1)
    if encoder_num_hiddens % encoder_num_heads != 0:
        encoder_num_hiddens += encoder_num_heads - (encoder_num_hiddens % encoder_num_heads)
        if encoder_num_hiddens % 2 != 0:
            encoder_num_hiddens += encoder_num_heads
            encoder_num_heads  += 1
    encoder_num_blocks      = max(int(parent.encoder_num_blocks * (2 * random.random())), 1)
    encoder_dropout         = min(max(numpy.random.normal(parent.encoder_dropout, scale=0.1), 0), 1)
    encoder_use_bias = \
        parent.encoder_use_bias if random.random() < 0.8 else not parent.encoder_use_bias

    decoder_num_hiddens     = max(int(parent.decoder_num_hiddens * (2 * random.random())), 2)
    if decoder_num_hiddens % 2 != 0:
        decoder_num_hiddens += 1
    decoder_ffn_num_hiddens = max(int(parent.decoder_ffn_num_hiddens * (2 * random.random())), 1)
    decoder_num_heads       = max(int(parent.decoder_num_heads * (2 * random.random())), 1)
    if decoder_num_hiddens % decoder_num_heads != 0:
        decoder_num_hiddens += decoder_num_heads - (decoder_num_hiddens % decoder_num_heads)
        if decoder_num_hiddens % 2 != 0:
            decoder_num_hiddens += decoder_num_heads
            decoder_num_heads  += 1
    decoder_num_blocks      = max(int(parent.decoder_num_blocks * (2 * random.random())), 1)
    decoder_dropout         = min(max(numpy.random.normal(parent.decoder_dropout, scale=0.1), 0), 1)

    first_phase_learning_rate  = parent.first_phase_learning_rate
    second_phase_learning_rate = parent.second_phase_learning_rate

    child = Configuration(batch_size,
                          encoder_num_hiddens,
                          encoder_ffn_num_hiddens,
                          encoder_num_heads,
                          encoder_num_blocks,
                          encoder_dropout,
                          encoder_use_bias,

                          decoder_num_hiddens,
                          decoder_ffn_num_hiddens,
                          decoder_num_heads,
                          decoder_num_blocks,
                          decoder_dropout,

                          first_phase_learning_rate,
                          second_phase_learning_rate,)
    return child


def evaluate(vocab: Vocabulary,
             languages: List[str],
             device: torch.device,
             max_length: int,
             validation_split: float,
             configuration: Configuration) -> float:
    """
    Given a configuration, return its validation loss.
    """

    data = read(languages, max_length=100, vocab=vocab)

    def encoder_instantiator():
        return Encoder(len(vocab),
                       configuration.encoder_num_hiddens,
                       configuration.encoder_ffn_num_hiddens,
                       configuration.encoder_num_heads,
                       configuration.encoder_num_blocks,
                       configuration.encoder_dropout,
                       max_length,
                       configuration.encoder_use_bias)
    def decoder_instantiator():
        return Decoder(len(vocab),
                       configuration.decoder_num_hiddens,
                       configuration.decoder_ffn_num_hiddens,
                       configuration.decoder_num_heads,
                       configuration.decoder_num_blocks,
                       configuration.decoder_dropout,
                       max_length)

    _, validation_losses_by_target_language = train(encoder_instantiator,
                                                    decoder_instantiator,
                                                    data,
                                                    configuration.first_phase_learning_rate,
                                                    configuration.second_phase_learning_rate,
                                                    validation_split,
                                                    configuration.batch_size,
                                                    max_length,
                                                    device)

    # we take the geometric mean of the validation losses.
    # this is not necessarily the best measurement, but it makes sense.
    # we can change this later if we think of a better way.

    overall = math.prod(validation_losses_by_target_language.values())
    return overall


def main() -> None:
    """
    Main function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam_size",
        type=int,
        required=True,
        help="The number of configurations to consider in the beams after each pass."
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        required=True,
        help="How many children of each configuration to consider during a pass."
    )
    parser.add_argument(
        "--passes",
        type=int,
        required=True,
        help="The number of beams passes to perform."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="The learning rate to use while training the model."
    )
    parser.add_argument(
        "--languages",
        type=str,
        required=True,
        help="The languages to use.",
    )
    args = parser.parse_args()

    seed_configuration = Configuration(batch_size=64,
                                       encoder_num_hiddens=12,
                                       encoder_ffn_num_hiddens=4,
                                       encoder_num_heads=3,
                                       encoder_num_blocks=2,
                                       encoder_dropout=0.5,
                                       encoder_use_bias=False,
                                       decoder_num_hiddens=12,
                                       decoder_ffn_num_hiddens=4,
                                       decoder_num_heads=2,
                                       decoder_num_blocks=3,
                                       decoder_dropout=0.5,
                                       first_phase_learning_rate=0.01,
                                       second_phase_learning_rate=0.01)
    beam = [seed_configuration]

    max_length = 100
    validation_split = 0.2
    vocab = Vocabulary('data/vocabulary.vocab')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _ in range(args.passes):
        beam = beam_search_pass(beam,
                                descend,
                                evaluate,
                                vocab,
                                args.languages.split(' '),
                                args.beam_size,
                                args.expansion_factor,
                                device,
                                max_length,
                                validation_split,)

    print('The best configuration is', beam[0])


if __name__ == "__main__":
    main()
