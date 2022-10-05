import argparse
import pickle
import torch

from vocab import Vocabulary
from typing import Dict, Generator, TextIO, List, Tuple


def tensors_from_pairs_file(file: TextIO, vocab: Vocabulary) -> \
    Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Extracts the sentences from a pairs file and yields them as tensors.
    """

    while True:
        try:
            sentence_1, sentence_2, _ = next(file), next(file), next(file)
        except StopIteration:
            break
        tensor_1 = torch.Tensor(vocab.sparsely_encoded(sentence_1))
        tensor_2 = torch.Tensor(vocab.sparsely_encoded(sentence_2))
        yield (tensor_1, tensor_2)

    file.close()


def read(languages: List[str], vocab: Vocabulary) \
    -> Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor], None, None]]:
    """
    Reads the data for the supplied languages and returns generators of the sentence pairs.
    The sentences are sparsely encoded.

    Arguments:
        languages: A list of languages to read.
                   The names should be given in the same format as the saved files in data/pairs
                   (e.g. 'es', 'fr', etc.)
        vocab: The vocabulary for all the languages.

    Returns:
        A dictionary mapping language pairs to generators of their sentences.
    """

    data: Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor], None, None]] = dict()
    for i, language_1 in enumerate(languages):
        for language_2 in languages[i+1:]:
            try:
                file = open(f'data/pairs/{language_1}-{language_2}.pairs', 'r')
                data[(language_1, language_2)] = tensors_from_pairs_file(file, vocab)
            except FileNotFoundError:
                file = open(f'data/pairs/{language_2}-{language_1}.pairs', 'r')
                data[(language_2, language_1)] = tensors_from_pairs_file(file, vocab)

    return data


def main() -> None:
    """
    Load the data for a given set of languages and print out a snippet of the sentence pairs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--languages',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    vocab = pickle.load(open('data/vocabulary.pickle', 'rb'))
    data = read(args.languages.split(' '), vocab)

    for pair, generator in data.items():
        print(pair, next(generator))

if __name__ == '__main__':
    main()
