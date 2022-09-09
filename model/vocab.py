import nltk

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
