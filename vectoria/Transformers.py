'''
Scikit Learn style vectorization of text.
'''
import html

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .FastTextLanguageModel import FastTextLanguageModel


# pylint: disable=invalid-name

class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """
    This class encapsulates treating text as vectors for use in machine
    learning, providing dense vectors extracted from FastText.

    This differs from the classic bag-of-words model in that context and
    word ordering a preserved, and subword ngram information is used to
    create vectors for out-of-vocabulary words.

    Attributes
    ----------
    language : The FastTextLanguageModel containing embeddings.
    maxlen : Limit to this number of words parsed per document.
    """

    def __init__(self, maxlen=1024, language='en'):
        """
        Load up and prepare the language model to transform words.

        Parameters
        ----------
        maxlen : int
            Limit parsing to this number of words.
        language : string, two character language code
            Specify the language model for SpaCy.

        """
        self.maxlen = maxlen
        self.language = FastTextLanguageModel(language)
        self.dimensions = self.language['hello'].shape[0]

    def fit(self, X, y=None, **kwargs):
        """
        Nothing to fit, this is a pre-trained set of vectors.
        """
        pass

    def transform(self, strings):
        """
        Transform an iterable source of strings into a dense embedding
        of FastText vectors.

        Parameters
        ----------
        strings : iterable
            An iterable of source strings to vectorize.

        Returns
        -------
        np.ndarray
            A 3 dimensional [string, word, embeddings] array, dense encoded.
        """
        if isinstance(strings, str):
            # forgive, otherwise this is just a HUGE memory leak
            strings = [strings]
        else:
            # buffering up the iterable source
            strings = list(strings)
        buffer = np.zeros((len(strings), self.maxlen,
                           self.dimensions), dtype=np.float32)
        for i, string in enumerate(strings):
            for j, word in enumerate(html.unescape(string)):
                buffer[i, j] = self.language[word]
        return buffer
