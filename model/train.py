import argparse

from collections import defaultdict
from typing import Dict, List, Iterable, Tuple
from vocab import Vocabulary


def train(learning_rate: float) -> Tuple[List[float], List[float]]:
    """
    Trains the model.

    :param learning_rate: The learning rate.
    :return: A list of the train loss values and a list of the validation loss values.
    """

    raise NotImplementedError


def tensors_from_pairs_file(file: Iterable[str], vocab_1, vocab_2) -> None:
    """
    Extracts the sentences from a pairs file and yields them as tensors.

    :param file: A stream of the contents of the pairs file. Each value is a line.
    :param vocab_1: The vocabulary for language 1. This method grows the vocabulary while reading.
    :param vocab_2: The vocabulary for language 2. This method grows the vocabulary while reading.
    :return: None.
    """

    while True:
        try:
            sentence_1, sentence_2, _ = next(file), next(file), next(file)
        except StopIteration:
            return 
        vocab_1.add_tokens_from_text(sentence_1)
        vocab_2.add_tokens_from_text(sentence_2)
        tensor_1 = vocab_1.sparsely_encoded(sentence_1)
        tensor_2 = vocab_2.sparsely_encoded(sentence_2)
        yield (tensor_1, tensor_2)


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
        '--learning_rate',
        type=float,
        required=False,
        default=0.001,
    )

    args = parser.parse_args()

    # we load in the dataset

    vocabularies: Dict[str, Vocabulary] = defaultdict(Vocabulary)
    languages = args.languages.split(' ')
    for i, language_1 in enumerate(languages):
        for language_2 in languages[i+1:]:
            try:
                with open(f'data/pairs/{language_1}-{language_2}.pairs', 'r') as file:
                    print(list(tensors_from_pairs_file(file, vocabularies[language_1], vocabularies[language_2])))
            except FileNotFoundError:
                with open(f'data/pairs/{language_2}-{language_1}.pairs', 'r') as file:
                    tensors_from_pairs_file(file, vocabularies[language_2], vocabularies[language_1])

    train(**args)


if __name__ == '__main__':
    main()