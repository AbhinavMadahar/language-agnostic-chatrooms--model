import argparse
from importlib.metadata import requires
import pickle
import torch

from vocab import Vocabulary, PAD
from typing import Dict, Generator, TextIO, List, Tuple


def tensors_from_pairs_file(file: TextIO, vocab: Vocabulary, max_length: int) -> \
    Generator[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]], None, None]:
    """
    Extracts the sentences from a pairs file and yields them as tensors along with their lengths.
    """

    while True:
        try:
            sentence_1, sentence_2, _ = next(file), next(file), next(file)
        except StopIteration:
            break

        sparsely_encoded_1 = vocab.sparsely_encoded(sentence_1)
        sparsely_encoded_2 = vocab.sparsely_encoded(sentence_2)

        sparsely_encoded_1 = sparsely_encoded_1[:max_length]
        sparsely_encoded_2 = sparsely_encoded_2[:max_length]

        length_1 = len(sparsely_encoded_1)
        length_2 = len(sparsely_encoded_2)

        pad_token = vocab.token_to_index[PAD]
        padding_1 = [pad_token] * (max_length - length_1)
        padding_2 = [pad_token] * (max_length - length_2)

        tensor_1 = torch.tensor(sparsely_encoded_1 + padding_1, requires_grad=True, dtype=torch.float)
        tensor_2 = torch.tensor(sparsely_encoded_2 + padding_2, requires_grad=True, dtype=torch.float)

        yield (tensor_1, tensor_2), (length_1, length_2)

    file.close()


def read(languages: List[str], max_length: int, vocab: Vocabulary | None = None) \
    -> Dict[
        Tuple[str, str],
        Generator[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]], None, None]]:
    """
    Reads the data for the supplied languages and returns generators of the sentence pairs and
    their lengths. The sentences are sparsely encoded.

    Arguments:
        languages: A list of languages to read.
                   The names should be given in the same format as the saved files in data/pairs
                   (e.g. 'es', 'fr', etc.)
        vocab: The vocabulary for all the languages.

    Returns:
        A dictionary mapping language pairs to generators of their sentences and their lengths.
    """

    if vocab is None:
        vocab = Vocabulary('data/vocabulary.vocab')

    data: Dict[
        Tuple[str, str],
        Generator[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]], None, None]
    ] = dict()
    for i, language_1 in enumerate(languages):
        for language_2 in languages[i+1:]:
            try:
                file = open(f'data/pairs/{language_1}-{language_2}.pairs', 'r')
                data[(language_1, language_2)] = tensors_from_pairs_file(file, vocab, max_length)
            except FileNotFoundError:
                file = open(f'data/pairs/{language_2}-{language_1}.pairs', 'r')
                data[(language_2, language_1)] = tensors_from_pairs_file(file, vocab, max_length)

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
