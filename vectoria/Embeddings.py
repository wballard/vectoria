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
        with open(path.with_suffix('.tmp'), 'wb') as f:
            chunk = 32 * 1024
            progress = tqdm(total=total_size, unit='B', unit_scale=True)
            for data in r.iter_content(chunk):
                if data:
                    f.write(data)
                    progress.update(len(data))
        path.with_suffix('.tmp').rename(path)


class WordEmbedding:
    """
    Language model based on word level parsing, encoding into pretrained Glove vectors.

    Attributes
    ----------
    embeddings
        A two dimensional numpy array [term id, vector dimension] storing floating points.
        This is a memory mapped array to save some I/O.

    >>> from vectoria import Embeddings
    >>> word = Embeddings.WordEmbedding(language='en')
    >>> word.embeddings.shape
    (1048576, 300)
    >>> word.embed('hello world')[0][0:4, 0:10]
    array([[  9.57600027e-02,  -3.96219999e-01,  -1.89219993e-02,
              2.85719991e-01,   4.91470009e-01,   3.94629985e-02,
              1.67980000e-01,  -1.49849996e-01,   2.31999997e-02,
              7.41180003e-01],
           [  5.22070006e-02,   1.64059997e-01,   1.42199993e-01,
             -1.41399996e-02,  -4.51810002e-01,  -3.03479992e-02,
              3.60689998e-01,  -3.12020004e-01,   7.19200005e-04,
              1.98919997e-01],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00]], dtype=float32)
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

        # a keras model to perform the actual embedding
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            mask_zero=True,
            input_length=maxlen,
            trainable=False,
            weights=[self.embeddings]))

    def embed(self, strings):
        """
        Given a string, turn it into a sequence of word identifiers, and
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
        embedded = self.model(input)
        model = keras.models.Model(input=input, output=embedded)
        return model.predict(self.sequencer.transform(strings))


class CharacterTrigramEmbedding:
    """
    Language model base that will download and compile pretrained FastText vectors
    for a given language.

    Attributes
    ----------
    embeddings
        A two dimensional numpy array [term id, vector dimension] storing floating points.
        This is a memory mapped array to save some I/O.

    >>> from vectoria import Embeddings
    >>> chargram = Embeddings.CharacterTrigramEmbedding(language='en')
    >>> chargram.embeddings.shape
    (196608, 300)
    >>> chargram.embed('hello')[0][0:4, 0:10]
    array([[ -4.47659999e-01,  -3.63579988e-01,  -3.11529994e-01,
              2.96270013e-01,   2.28880003e-01,  -1.85499996e-01,
             -8.03470016e-02,  -3.20030004e-02,  -8.14009979e-02,
             -3.94560009e-01],
           [ -4.00400013e-01,   3.42779997e-04,  -1.96740001e-01,
              3.76879990e-01,  -2.82209992e-01,  -2.56969988e-01,
             -1.47990003e-01,   2.39020005e-01,   1.29620001e-01,
             -2.53360003e-01],
           [ -1.79299995e-01,  -1.67520002e-01,  -3.27329993e-01,
              2.04939991e-02,  -3.84220004e-01,  -9.49349999e-02,
             -1.96329996e-01,   4.60180014e-01,   2.26310000e-01,
             -3.08160007e-01],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00]], dtype=float32)
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
            with open(vectors_path, 'r') as f:
                first_line = f.readline()
                words, dimensions = map(int, first_line.split())
                embeddings = np.memmap(final_path.with_suffix(
                    '.tmp'), dtype='float32', mode='w+', shape=(sequencer.features, dimensions))
            for line in tqdm(iterable=open(str(vectors_path)), total=words, desc='Parsing', unit='vector'):
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
        with open(vectors_path, 'r') as f:
            first_line = f.readline()
            words, dimensions = map(int, first_line.split())
            self.embeddings = np.memmap(
                final_path, dtype='float32', mode='r', shape=(sequencer.features, dimensions))

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            mask_zero=True,
            input_length=maxlen,
            trainable=False,
            weights=[self.embeddings]))

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
        A two dimensional embedding array.
        """
        input = keras.layers.Input(shape=(self.maxlen,))
        embedded = self.model(input)
        model = keras.models.Model(input=input, output=embedded)
        return model.predict(self.sequencer.transform(strings))
