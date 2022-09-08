import argparse
import random
import torch

from collections import defaultdict
from data import read
from model import Encoder, Decoder
from torch import nn
from typing import Any, Dict, List, Generator, Iterable, Tuple
from vocab import Vocabulary


def train_many_to_many(encoder: Encoder,
                       decoder: Decoder, 
                       data: Iterable[Tuple[torch.Tensor]],
                       learning_rate: float, 
                       validation_split: float,
    ) -> Tuple[List[float], List[float]]:
    """
    Trains the model for the first phase.
    In this phase, we train the model to translate from every language to every other language.

    :param learning_rate: The learning rate.
    :return: A list of the train loss values and a list of the validation loss values.
    """

    raise NotImplementedError


def train_many_to_many_single_epoch(encoder: Encoder,
                                    decoder: Decoder,
                                    data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                                    learning_rate: float,
                                    validation_split: float,
    ) -> Tuple[List[float], List[float]]:

    raise NotImplementedError


def train_many_to_one(encoder: Encoder,
                      decoder: Decoder,
                      data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                      learning_rate: float,
                      validation_split: float,
    ) -> Tuple[List[float], List[float]]:
    """
    Trains the model for the second phase.
    In this phase, we fine-tune the general, many-to-many model to translate to a specific language.

    :param learning_rate: The learning rate.
    :return: A list of the train loss values and a list of the validation loss values.
    """

    raise NotImplementedError


def train_many_to_many_single_epoch(encoder: Encoder,
                                    decoder: Decoder,
                                    data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                                    learning_rate: float,
                                    validation_split: float,
    ) -> Tuple[List[float], List[float]]:

    raise NotImplementedError


def train(encoder: Encoder,
          decoder: Decoder,
          data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
          first_phase_learning_rate: int,
          second_phase_learning_rate: int,
          validation_split: float,
    ) -> Dict[str, Tuple[Encoder, Decoder]]:
    """
    Trains the model over both phases.
    It returns the many-to-one machine translation models.

    :param encoder: The encoder.
    :param decoder: The decoder.
    :param data: A dictionary which maps language pairs (e.g. ('en', 'fr')) to an iterator of sentence pairs tensors.
    :param first_phase_learning_rate: The learning rate for the first phase.
    :param second_phase_learning_rate: The learning rate for the second phase.
    :return: A dictionary mapping languages to their many-to-one machine translation models.
    """

    def randomly_sampled_sentence_pairs(data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]]) -> Generator[Tuple[torch.Tensor, torch.Tensor]]:
        "Use the existing iterables to make a new iterable which randomly samples with uniformity over the language pairs."
        language_pairs_which_still_have_data = list(data.keys())
        while len(language_pairs_which_still_have_data) != 0:
            language_pair = random.choice(language_pairs_which_still_have_data)
            try:
                tensor_1, tensor_2 = next(data[language_pair])
                yield tensor_1, tensor_2
                yield tensor_2, tensor_1
            except StopIteration:
                language_pairs_which_still_have_data.remove(language_pair)
    
    def randomly_sampled_sentence_pairs_for_single_language_pair(data: Dict[Tuple[str, str], Iterable[Tuple[torch.Tensor, torch.Tensor]]],
                                                                 language: str
        ) -> Generator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tensors which translate from any language to the given language.
        This should be used in the second phase.
        """

        pairs_involving_language_which_still_have_data = [pair for pair in data.keys() if language in pair]
        while len(pairs_involving_language_which_still_have_data) != 0:
            language_pair = random.choice(pairs_involving_language_which_still_have_data)
            try:
                tensor_1, tensor_2 = next(data[language_pair])
                yield tensor_1, tensor_2 if language_pair[1] == language else tensor_2, tensor_1
            except StopIteration:
                pairs_involving_language_which_still_have_data.remove(language_pair)
        
    # phase 1
    train_many_to_many(encoder, decoder, randomly_sampled_sentence_pairs(data), learning_rate=first_phase_learning_rate, validation_split=validation_split)

    # phase 2
    languages = []
    for pair in data.keys():
        languages.append(pair[0])
        languages.append(pair[1])
    
    many_to_one_models: Dict[str, Tuple[Encoder, Decoder]] = dict()
    base_encoder, base_decoder = encoder, decoder
    for language in languages:
        encoder, decoder = clone(base_encoder), clone(base_decoder)
        train_many_to_one(encoder, decoder, randomly_sampled_sentence_pairs_for_single_language_pair(data, language), second_phase_learning_rate, validation_split=validation_split)
        many_to_one_models[language] = (encoder, decoder)
    
    return many_to_one_models


def clone(model: nn.Module) -> nn.Module:
    """
    Clone a model.
    
    :param model: The model to clone.
    :return: The deep copy of the model.
    """

    raise NotImplementedError


def main() -> None:
    """
    Given a hyperparameter configuration, train the model using that configuration.
    The configuration is given by the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--languages',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--first_phase_learning_rate',
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        '--second_phase_learning_rate',
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        required=False,
        default=0.1,
    )

    args = parser.parse_args()

    # we load in the dataset

    encoder = Encoder()
    decoder = Decoder()

    train(encoder, decoder, read(args.languages.split(' ')), args.first_phase_learning_rate, args.second_phase_learning_rate, args.validation_split)


if __name__ == '__main__':
    main()
