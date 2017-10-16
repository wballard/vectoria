"""
Turns strings into sequences of ordinal integers, intended to be used
with further embeddings.

These follow the sklearn style, with no need to `fit`, just `transform`
as hashing is used.
"""
import html

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import mmh3


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
            for j, word in enumerate(self.build_analyzer()(html.unescape(string))):
                if j >= self.maxlen:
                    break
                else:
                    buffer[i, j] = abs(mmh3.hash(word) % self.features)
        return buffer


class WordSequencer(HashingTransformMixin, CountVectorizer):
    """
    Treat text as a sequence of words.

    This differs from the classic bag-of-words model in that context of
    word ordering a preserved.


    Attributes
    ----------
    maxlen : Limit to this number of words parsed per document.
    features: Total number of unique features, which may allow collisions.

    >>> from vectoria import Sequencers
    >>> Sequencers.WordSequencer(maxlen=3).transform(['hello world'])
    array([[784967, 408827,      0]], dtype=int32)
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


class CharacterTrigramSequencer(HashingTransformMixin, CountVectorizer):
    """
    Treat text as a sequence of character trigrams.

    This differs from the classic bag-of-words model in that context of
    word ordering a preserved.

    Attributes
    ----------
    maxlen:
        Limit to this number of chargrams parsed per document.
    features: 
        Total number of unique features, which may allow collisions.

    >>> from vectoria import Sequencers
    >>> Sequencers.CharacterTrigramSequencer(maxlen=4).transform(['hello'])
    array([[ 12358, 130791,  85660,      0]], dtype=int32)
    """

    def __init__(self, maxlen=1024):
        """
        Configure word analysis for character trigrams.

        Parameters
        ----------
        maxlen : int
            Limit parsing to this number of words.
        """
        super(CharacterTrigramSequencer, self).__init__(
            lowercase=True, ngram_range=(3, 3), analyzer='char')
        self.maxlen = maxlen
        self.features = (2**16) * 3  # 2 ** 16 for unicode, * 3 for trigram


class SubwordSequencer(CharacterTrigramSequencer):
    """
    To support FastText type encodings, treat text as a series of words,
    and then break those words into character ngram subwords.

    Attributes
    ----------
    maxwords:
        Limit to this number of words.
    maxngrams:
        For each word limit to this number of ngrams.
    features:
        Total number of unique features, which may allow collisions.
    wordbreaker:
        Split an initial string into words.

    >>> from vectoria import Sequencers
    >>> Sequencers.SubwordSequencer(maxwords=4, maxngrams=6).transform(['hello world'])
    array([[[ 12358, 130791,  85660,      0,      0,      0],
            [156642, 179925, 179940,      0,      0,      0],
            [     0,      0,      0,      0,      0,      0],
            [     0,      0,      0,      0,      0,      0]]], dtype=int32)
    """

    def __init__(self, maxwords=1024, maxngrams=32):
        """
        Configure word analysis for character trigrams.

        Parameters
        ----------
        maxlen : int
            Limit parsing to this number of words.
        maxngrams:
            For each word limit to this number of ngrams.
        """
        super(SubwordSequencer, self).__init__(maxngrams)
        self.maxngrams = maxngrams
        self.maxwords = maxwords
        self.wordbreaker = CountVectorizer(lowercase=True).build_analyzer()

    def transform(self, strings):
        """
        Given a string, or iterable source of strings, break into words,
        then subwords by character ngram. Each word is thus represented by multiple
        identifiers.

        Parameters
        ----------
        strings : iterable
            An iterable of source strings to vectorize.

        Returns
        -------
        np.ndarray
            A 3 tensor [string_index, word_index, subword_index], with a 32 bit word identifier.
        """
        if isinstance(strings, str):
            # forgive, otherwise this is just going character wise
            strings = [strings]
        else:
            # buffering up the iterable source
            strings = list(strings)
        buffer = np.zeros((len(strings), self.maxwords,
                           self.maxngrams), dtype=np.int32)
        ngram_maker = super(SubwordSequencer, self).transform
        for i, string in enumerate(strings):
            ngrams = ngram_maker(self.wordbreaker(string))
            buffer[i, :ngrams.shape[0]] = ngrams
        return buffer
