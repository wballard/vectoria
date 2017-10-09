'''
Turns strings into sequences of ordinal integers, intended to be used
with further embeddings.

These follow the sklearn style, with no need to `fit`, just `transform`
as hashing is used.
'''
import html

import numpy as np
import pyhash
from sklearn.feature_extraction.text import CountVectorizer


class HashingTransformMixin:
    """
    Hashing, counts on `self.maxlen` and `self.features`.
    """

    def fit(self, strings, **kwargs):
        """
        Nothing to fit, this will be using hashing.
        """
        pass

    def transform(self, strings):
        """
        Transform an iterable source of strings into a dense matrix
        of ordered identifiers, up to `self.maxlen`.

        Zeros are placeholders, and can be used as a mask if needed.

        Parameters
        ----------
        strings : iterable
            An iterable of source strings to vectorize.

        Returns
        -------
        np.ndarray
            A 2 dimensional [string_index, word_index], with a 32 bit word identifier.
        """
        if isinstance(strings, str):
            # forgive, otherwise this is just going character wise
            strings = [strings]
        else:
            # buffering up the iterable source
            strings = list(strings)
        buffer = np.zeros((len(strings), self.maxlen), dtype=np.int32)
        for i, string in enumerate(strings):
            for j, word in enumerate(self.build_analyzer(html.unescape(string))):
                if j >= self.maxlen:
                    break
                else:
                    buffer[i, j] = pyhash.murmur3_32(word) % self.features
        return buffer


class WordSequencer(CountVectorizer, HashingTransformMixin):
    """
    Treat text as a sequence of words.

    This differs from the classic bag-of-words model in that context of
    word ordering a preserved.

    Attributes
    ----------
    maxlen : Limit to this number of words parsed per document.
    features: Total number of unique features, which may allow collisions.
    """

    def __init__(self, maxlen=1024):
        """
        Load up and prepare the language model to transform words.

        Parameters
        ----------
        maxlen : int
            Limit parsing to this number of words.
        """
        super(WordSequencer, self).__init__(lowercase=True)
        self.maxlen = maxlen
        self.features = 2 ** 20


class CharacterTrigramSequencer(CountVectorizer):
    """
    Treat text as a sequence of character trigrams.

    This differs from the classic bag-of-words model in that context of
    word ordering a preserved.

    Attributes
    ----------
    maxlen : Limit to this number of chargrams parsed per document.
    """

    def __init__(self, maxlen=1024):
        """
        Load up and prepare the language model to transform words.

        Parameters
        ----------
        maxlen : int
            Limit parsing to this number of words.
        """
        super(WordSequencer, self).__init__(lowercase=True, ngram_range=(3, 3))
        self.maxlen = maxlen
        self.features = (2**16) * 3  # 2 ** 16 for unicode, * 3 for trigram
