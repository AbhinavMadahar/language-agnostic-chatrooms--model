import argparse
import torch

from collections import defaultdict
from vocab import Vocabulary
from typing import Dict, Generator, Iterable, List, Tuple


def vocab_from_pairs_file(filename: str, language: str, vocab: Vocabulary = None) -> Vocabulary:
    if vocab is None:
        vocab = Vocabulary(language)
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(language):
                without_prefix = line[len(language)+2:]  # we skip the language code, the colon, and the space
                print(without_prefix)
                vocab.add_tokens_from_text(without_prefix)
    
    return vocab


def tensors_from_pairs_file(file: Iterable[str], vocab_1: Vocabulary, vocab_2: Vocabulary) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Extracts the sentences from a pairs file and yields them as tensors.

    :param file: A stream of the contents of the pairs file. Each value is a line.
    :param vocab_1: The vocabulary for language 1. This method grows the vocabulary while reading.
    :param vocab_2: The vocabulary for language 2. This method grows the vocabulary while reading.
    """

    while True:
        try:
            sentence_1, sentence_2, _ = next(file), next(file), next(file)
        except StopIteration:
            break 
        tensor_1 = vocab_1.sparsely_encoded(sentence_1)
        tensor_2 = vocab_2.sparsely_encoded(sentence_2)
        yield (tensor_1, tensor_2)
    
    file.close()


def read(languages: List[str]) -> Tuple[Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor], None, None]], Dict[str, Vocabulary]]:
    """
    Reads the data for the supplied languages and returns the sentence pairs along with the vocabularies for the languages.
    
    Arguments:
        languages: A list of languages to read.
                   The names should be given in the same format as the saved files in data/pairs (e.g. 'es', 'fr', etc.)
    
    Returns:
        A dictionary mapping language pairs to generators of their sentences and a dictionary mapping languages to their vocabularies.
    """
    
    vocabularies: Dict[str, Vocabulary] = defaultdict(Vocabulary)

    data: Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor]]] = dict()
    for i, language_1 in enumerate(languages):
        for language_2 in languages[i+1:]:
            try:
                file = open(f'data/pairs/{language_1}-{language_2}.pairs', 'r')
                data[(language_1, language_2)] = tensors_from_pairs_file(file, vocabularies[language_1], vocabularies[language_2])
            except FileNotFoundError:
                file = open(f'data/pairs/{language_2}-{language_1}.pairs', 'r')
                data[(language_2, language_1)] = tensors_from_pairs_file(file, vocabularies[language_2], vocabularies[language_1])
    
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

    data = read(args.languages.split(' '))

    for pair, generator in data.items():
        print(pair, next(generator))

if __name__ == '__main__':
    main()
