import torch

from collections import defaultdict
from vocab import Vocabulary
from typing import Dict, Generator, Iterable, List, Tuple


def tensors_from_pairs_file(file: Iterable[str], vocab_1, vocab_2) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
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
            return 
        vocab_1.add_tokens_from_text(sentence_1)
        vocab_2.add_tokens_from_text(sentence_2)
        tensor_1 = vocab_1.sparsely_encoded(sentence_1)
        tensor_2 = vocab_2.sparsely_encoded(sentence_2)
        yield (tensor_1, tensor_2)


def read(languages: List[str]) -> Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor], None, None]]:
    vocabularies: Dict[str, Vocabulary] = defaultdict(Vocabulary)

    data: Dict[Tuple[str, str], Generator[Tuple[torch.Tensor, torch.Tensor]]] = dict()
    for i, language_1 in enumerate(languages):
        for language_2 in languages[i+1:]:
            try:
                with open(f'data/pairs/{language_1}-{language_2}.pairs', 'r') as file:
                    data[(language_1, language_2)] = tensors_from_pairs_file(file, vocabularies[language_1], vocabularies[language_2])
            except FileNotFoundError:
                with open(f'data/pairs/{language_2}-{language_1}.pairs', 'r') as file:
                    data[(language_2, language_1)] = tensors_from_pairs_file(file, vocabularies[language_2], vocabularies[language_1])
    
    return data
