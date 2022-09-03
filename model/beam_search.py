import argparse
import random

from collections import namedtuple
from typing import Callable, List


Configuration = namedtuple('Configuration', ['learning_rate'])


def beam_search_pass(beam: List[Configuration], 
                     descend: Callable[[Configuration], Configuration],
                     evaluate: Callable[[Configuration], float],
                     beam_size: int,
                     expansion_factor: int,
    ) -> List[Configuration]:
    """
    Given a list of configurations, performs a beam search pass.

    :param beam: An list of configurations.
    :param beam_size: How many configurations to return.
    :param descend: A function which takes a configuration and returns a similar configuration.
    :param evaluate: A function which takes a configuration and returns how bad it is.
                     A good idea is to return the final validation loss after training using that configuration.
    :param expansion_factor: How many children to consider per configuration.
    :return: A list of the best configurations from the pass.
             The configurations are sorted in descending order, so the 0th configuration is the best.
    """

    children = [descend(configuration) for _ in range(expansion_factor) for configuration in beam]
    children = sorted(children, key=evaluate)
    children = children[:beam_size]

    return children


def descend(configuration: Configuration) -> Configuration:
    """
    Given a configuration, find a similar configuration.

    :param configuration: A configuration.
    :return: A similar configuration.
    """

    learning_rate = configuration.learning_rate * (2 * random.random())

    return Configuration(learning_rate)


def evaluate(configuration: Configuration) -> float:
    """
    Given a configuration, return its validation loss.
    """

    raise NotImplementedError


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
    args = parser.parse_args()

    seed_configuration = Configuration(args.learning_rate)
    beam = [seed_configuration]
    print(beam)

    for _ in range(args.passes):
        beam = beam_search_pass(beam, descend, evaluate, args.beam_size, args.expansion_factor)
        print(beam)
    
    return beam[0]


if __name__ == "__main__":
    main()