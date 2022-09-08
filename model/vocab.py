import nltk

from typing import Dict, List


OOV = 0  # out of vocabulary
EOS = 1  # end of sequence

class Vocabulary:
    """
    Handle the vocabulary of a language.
    """
    def __init__(self, language_name: str = None) -> None:
        self.language_name = language_name
        self.token_to_index: Dict[str, int] = {OOV: 0, EOS: 1}
        self.index_to_token: Dict[str, int] = {index:token for token, index in self.token_to_index.items()}
    
    def add_token(self, token: str) -> None:
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
    
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
            if token in self.token_to_index:
                encoded.append(self.token_to_index[token])
            else:
                encoded.append(self.token_to_index[OOV])
        encoded.append(self.token_to_index[EOS])

        return encoded
