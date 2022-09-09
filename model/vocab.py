"""
Manipulate vocabularies.

This module can be used to generate a vocabulary from a sentences file.
To do so, give the name of the language.
The program will save the vocabulary to the data/vocabularies directory in a file called language_name.vocab
"""

import argparse
import nltk
import pickle
import os

from collections import Counter
from typing import Dict, List


OOV = 0  # out of vocabulary
EOS = 1  # end of sequence
SOS = 2  # start of sequence

class Vocabulary:
    """
    Handle the vocabulary of a language.
    """
    def __init__(self, language_name: str = None, min_frequency: int = None) -> None:
        """
        Initialize the vocabulary.

        Args:
            language_name (str, optional): name of the language. Defaults to None.
            min_frequency (int, optional): if a token appears less than this many times while building the vocabulary, consider it OOV.
                                           If None, then no words are treated as OOV.
                                           Defaults to None.
        """
        self.language_name = language_name

        self.token_to_index: Dict[str, int] = dict()
        self.index_to_token: Dict[str, int] = {index:token for token, index in self.token_to_index.items()}

        self.frequency_of_token: Dict[str, int] = Counter()
        self.min_frequency = min_frequency

        self.add_token(OOV)
        self.add_token(EOS)
        self.add_token(SOS)
    
    def add_token(self, token: str) -> None:
        index = len(self)

        self.token_to_index[token] = index
        self.index_to_token[index] = token

        self.frequency_of_token[token] += 1
    
    def add_tokens_from_text(self, text: str) -> None:
        tokens = nltk.tokenize.wordpunct_tokenize(text)
        for token in tokens:
            self.add_token(token)
    
    def __contains__(self, token) -> bool:
        return token in self.token_to_index
    
    def __len__(self) -> int:
        return len(self.token_to_index)
    
    def sparsely_encoded(self, text: str) -> List[int]:
        "Encode into indices, e.g. 'I like dogs' might become [1, 45, 123]."

        encoded = []
        for token in nltk.tokenize.wordpunct_tokenize(text):
            if token in self.token_to_index and (self.min_frequency is None or self.frequency_of_token[token] >= self.min_frequency):
                encoded.append(self.token_to_index[token])
            else:
                encoded.append(self.token_to_index[OOV])
        encoded.append(self.token_to_index[EOS])

        return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="The name of the language."
    )
    args = parser.parse_args()
    
    language = args.language
    
    vocab = Vocabulary(language)
    with open(f'data/sentences/{language}.txt', 'r') as file:
        for sentence in file:
            vocab.add_tokens_from_text(sentence)

    if not os.path.exists('data/vocabularies'):
        os.makedirs('data/vocabularies')

    with open(f'data/vocabularies/{language}.vocab', 'wb') as file:
        pickle.dump(vocab, file)

if __name__ == '__main__':
    main()
