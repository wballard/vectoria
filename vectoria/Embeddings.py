"""
Download and store pretrained models that can be used to create embedding
layers.

These models take pretrained files, downloaded over HTTP, and compile them
into dense tensor representation using a memory mapping back end.

"""
import importlib
import zipfile
from pathlib import Path

import keras
import numpy as np
import numpy.linalg as la
import requests
from tqdm import tqdm

from . import Sequencers

FAST_TEXT_URL_TEMPLATE = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{0}.vec"
GLOVE_URL_EN = "http://nlp.stanford.edu/data/glove.6B.zip"


def download_path(flavor, language):
    """
    Compute a download path for a given flavor of vectors
    and language.

    Parameters
    ----------
    flavor:
        Any string to separate different models.
    language:
        Two letter language code.

    Returns
    -------
    A `Path` object.
    """
    # the local in package file path for the language model
    pkg = importlib.import_module('vectoria')
    vectoria_path = Path(pkg.__file__).parent
    folder_path = vectoria_path / Path(language)
    if not folder_path.exists():
        folder_path.mkdir()
    vectors_path = folder_path / Path('{1}-{0}.vec'.format(flavor, language))
    return vectors_path


def download_if_needed(url, path):
    """
    Download a file with progress, with a temporary file swap.

    This will not download if the path is already present / downloaded.

    Parameters
    ----------
    url:
        Source URL on network.
    path:
        Target Path on disk.
    """
    # download if needed
    if not path.exists():
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        with open(path.with_suffix('.tmp'), mode='wb') as f:
            chunk = 32 * 1024
            progress = tqdm(total=total_size, unit='B', unit_scale=True)
            for data in r.iter_content(chunk):
                if data:
                    f.write(data)
                    progress.update(len(data))
        path.with_suffix('.tmp').rename(path)


class Embedding:
    """
    Provides the core method to override to build a model, along with
    an embed method to exercise/test the embedding.

    Inherit from this to create different text embeddings.
    """

    def embed(self, strings):
        """
        Given a string, turn it into a sequence of chargram identifiers, and
        then embed it.

        Parameters
        ----------
        strings:
            Any string, or an array batch of strings

        Returns
        -------
        A three tensor, (batch entry, word position, embeded value).
        """
        input = keras.layers.Input(shape=(self.maxlen,))
        embedded = self.build_model()(input)
        model = keras.models.Model(inputs=input, outputs=embedded)
        return model.predict(self.sequencer.transform(strings))

    def build_model(self):
        """
        A simple keras model that takes direct advantage of the embedding layer.
        """
        # a keras model to perform the actual embedding
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            input_length=self.maxlen,
            trainable=False,
            weights=[self.embeddings])
        )
        return model


class WordEmbedding(Embedding):
    """
    Language model based on word level parsing, encoding into pretrained Glove vectors.

    Attributes
    ----------
    embeddings
        A two dimensional numpy array [term id, vector dimension] storing floating points.
        This is a memory mapped array to save some I/O.

    >>> from vectoria import Embeddings
    >>> word = Embeddings.WordEmbedding(language='en', maxlen=4)
    >>> word.embed('hello world')
    array([[[ 0.09576   , -0.39622   , -0.018922  , ...,  0.35997999,
             -0.030634  ,  0.010207  ],
            [ 0.052207  ,  0.16406   ,  0.14219999, ...,  0.32302999,
             -0.037163  , -0.048231  ],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ]]], dtype=float32)
    """

    def __init__(self, language='en', maxlen=1024):
        """
        Construct a language model for a given string by:
        - opening an existing model if present
        - downloading and compiling pretrained word models otherwise

        This is a mulit-gigabyte download at least for english and will take
        a while.


        Parameters
        ----------
        language:
            Two letter language code.
            Currently on english 'en' is supported.
        maxlen:
            Limit to this number of token parsed per document.
        """
        vectors_path = download_path('glove', language).with_suffix('.zip')
        final_path = vectors_path.with_suffix('.numpy')
        url = GLOVE_URL_EN
        # glove known to have 300 dimensions
        dimensions = 300
        self.maxlen = maxlen
        self.sequencer = sequencer = Sequencers.WordSequencer(maxlen=maxlen)
        download_if_needed(url, vectors_path)
        # compile if needed, mapping from each trigram available
        if not final_path.exists():
            with zipfile.ZipFile(vectors_path.as_posix()) as archive:
                words = 0
                for line in tqdm(archive.open('glove.6B.300d.txt'), desc='Counting', unit='vector'):
                    words = words + 1
                embeddings = np.memmap(final_path.with_suffix(
                    '.tmp'), dtype='float32', mode='w+', shape=(sequencer.features, dimensions))
                for line in tqdm(archive.open('glove.6B.300d.txt'), total=words, desc='Parsing', unit='vector'):
                    line = line.decode('utf8')
                    segments = line.split()
                    assert len(segments) == 301
                    word = sequencer.transform([segments[0]])[0]
                    try:
                        numbers = np.array(
                            list(map(np.float32, segments[1:])))
                        embeddings[word] = numbers
                    except ValueError:
                        pass
            # the zero word is a pad value
            embeddings[0] = np.zeros(dimensions)
            embeddings.flush()
            del embeddings
            final_path.with_suffix('.tmp').rename(final_path)
        # and -- actually open
        self.embeddings = np.memmap(
            final_path, dtype='float32', mode='r', shape=(sequencer.features, dimensions))


class CharacterTrigramEmbedding(Embedding):
    """
    Language model base that will download and compile pretrained FastText vectors
    for a given language.

    Attributes
    ----------
    embeddings
        A two dimensional numpy array [term id, vector dimension] storing floating points.
        This is a memory mapped array to save some I/O.

    >>> from vectoria import Embeddings
    >>> chargram = Embeddings.CharacterTrigramEmbedding(language='en', maxlen=6)
    >>> chargram.embed('hello')
    array([[[ -4.47659999e-01,  -3.63579988e-01,  -3.11529994e-01, ...,
              -5.76590002e-01,   2.53699988e-01,  -3.65200005e-02],
            [ -4.00400013e-01,   3.42779997e-04,  -1.96740001e-01, ...,
              -2.23260000e-01,  -2.42109999e-01,   1.57110006e-01],
            [ -1.79299995e-01,  -1.67520002e-01,  -3.27329993e-01, ...,
               6.78020000e-01,   4.57850009e-01,  -9.05459970e-02],
            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
               0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
               0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
               0.00000000e+00,   0.00000000e+00,   0.00000000e+00]]], dtype=float32)
    """

    def __init__(self, language='en', maxlen=1024):
        """
        Construct a language model for a given string by:
        - opening an existing model if present
        - downloading and compiling pretrained word models otherwise

        This is a mulit-gigabyte download at least for english and will take
        a while.

        Parameters
        ----------
        language:
            Two letter language code.
        maxlen:
            Limit to this number of token parsed per document.
        """
        vectors_path = download_path('fasttext', language)
        final_path = vectors_path.with_suffix('.numpy')
        url = FAST_TEXT_URL_TEMPLATE.format(language)
        self.maxlen = maxlen
        self.sequencer = sequencer = Sequencers.CharacterTrigramSequencer(
            maxlen=maxlen)
        download_if_needed(url, vectors_path)
        # compile if needed, mapping from each trigram available
        if not final_path.exists():
            with open(vectors_path, mode='r', encoding='utf8') as f:
                first_line = f.readline()
                words, dimensions = map(int, first_line.split())
                embeddings = np.memmap(final_path.with_suffix(
                    '.tmp'), dtype='float32', mode='w+', shape=(sequencer.features, dimensions))
            for line in tqdm(iterable=open(str(vectors_path), mode='r', encoding='utf8'), total=words, desc='Parsing', unit='vector'):
                # how big is this thing? We are only interested in trigrams
                segments = line.split()
                if len(segments) > dimensions and len(segments[0]) == 3:
                    word = sequencer.transform([segments[0]])[0][0]
                    try:
                        numbers = np.array(list(map(np.float32, segments[1:])))
                        embeddings[word] = numbers
                    except ValueError:
                        pass
            # the zero word is a pad value
            embeddings[0] = np.zeros(dimensions)
            embeddings.flush()
            del embeddings
            final_path.with_suffix('.tmp').rename(final_path)
        # and -- actually open
        with open(vectors_path, mode='r', encoding='utf8') as f:
            first_line = f.readline()
            words, dimensions = map(int, first_line.split())
            self.embeddings = np.memmap(
                final_path, dtype='float32', mode='r', shape=(sequencer.features, dimensions))


class FastTextEmbedding(CharacterTrigramEmbedding):
    """
    Encode 3 dimensional subword sequences using a FastText combination of
    ngram components.

    As an example assume we are encoding the word 'hello', the ngram components
    are:
    hel ell llo

    Each component ngram has a sequence identifier, and a corresponding dense embedding.
    To create an embedding for the entire word, simply combine the subword components
    with addition.
    
    >>> from vectoria import Embeddings
    >>> ft = Embeddings.FastTextEmbedding(language='en', maxwords=4, maxngrams=6)
    >>> ft.embed('hello world')
    array([[[-0.0605947 , -0.03130458, -0.04928451, ..., -0.00718565,
              0.02768803,  0.00177203],
            [ 0.01353295, -0.03353724,  0.00498307, ...,  0.02882224,
              0.03165274,  0.01352788],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ]]], dtype=float32)
    """

    def __init__(self, language='en', maxwords=1024, maxngrams=32):
        """
        maxwords:
            Limit to this number of words.
        maxngrams:
            For each word limit to this number of ngrams.
        """
        self.maxwords = maxwords
        self.maxngrams = maxngrams
        super(FastTextEmbedding, self).__init__(
            language=language, maxlen=maxngrams)
        self.sequencer = Sequencers.SubwordSequencer(
            maxwords=maxwords, maxngrams=maxngrams)

    def embed(self, strings):
        """
        Given a string, turn it into a sequence of chargram identifiers, and
        then embed it.

        Parameters
        ----------
        strings:
            Any string, or an array batch of strings

        Returns
        -------
        A three tensor, (batch entry, word position, embeded value).
        """
        input = keras.layers.Input(shape=(self.maxwords, self.maxngrams))
        embedded = self.build_model()(input)
        model = keras.models.Model(inputs=input, outputs=embedded)
        return model.predict(self.sequencer.transform(strings))

    def build_model(self):
        """
        A keras model that embeds, and then combines ngrams to form word embeddings.
        """
        # a keras model to perform the actual embedding
        model = keras.models.Sequential()
        embed = keras.layers.Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            mask_zero=True,
            input_length=self.maxngrams,
            trainable=False,
            embeddings_initializer=lambda shape: self.embeddings)
        model.add(keras.layers.TimeDistributed(embed, input_shape=(self.maxwords, self.maxngrams)))
        model.add(FastTextCombinator())
        return model


class FastTextCombinator(keras.layers.Layer):
    """
    A custom keras layer that reduces the embedding per ngram 
    down to an embedding per word.
    """

    def call(self, ngram_sequence_identifiers):
        """
        Sum the ngram embeddings and normalize.
        """
        combined_ngrams_to_words = keras.backend.sum(ngram_sequence_identifiers, axis=2)
        return keras.backend.l2_normalize(combined_ngrams_to_words)

    def compute_output_shape(self, input_shape):
        """
        The output shape removes the ngram dimension.
        """
        return (None, input_shape[1], input_shape[-1])
