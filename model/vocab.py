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
SOS = 1  # start of sequence
EOS = 2  # end of sequence

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

        self.token_to_index: Dict[str, int] = {OOV: 0, SOS: 1, EOS: 2}
        self.index_to_token: Dict[str, int] = {index:token for token, index in self.token_to_index.items()}

        self.frequency_of_token: Dict[str, int] = Counter()
        self.min_frequency = min_frequency
    
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
    
    def remove_uncommon(self) -> None:
        "Removes all tokens which appear fewer times than min_frequency."
        if self.min_frequency is None:
            return

        new_token_to_index: Dict[str, int] = {OOV: 0, SOS: 1, EOS: 2}
        for token, frequency in self.frequency_of_token.items():
            if frequency >= self.min_frequency:
                new_token_to_index[token] = len(new_token_to_index)
        
        self.token_to_index = new_token_to_index
        self.index_to_token: Dict[str, int] = {index:token for token, index in self.token_to_index.items()}
    
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
        help="The name of the language(s)."
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        required=False,
        default=None,
        help="The minimum number of times a token must appear for it to be saved in the vocabulary."
    )
    args = parser.parse_args()

    languages = args.language.split(' ')

    vocab = Vocabulary(min_frequency=args.min_frequency)

    # at the end of each input sequence, we specify which language to target
    for language in languages:
        vocab.add_token(f'<Target language: {language}>')

    for language in languages:
        with open(f'data/sentences/{language}.txt', 'r') as file:
            for sentence in file:
                vocab.add_tokens_from_text(sentence)

    vocab.remove_uncommon()

    with open(f'data/vocabulary.pickle', 'wb') as file:
        pickle.dump(vocab, file)

if __name__ == '__main__':
    main()
