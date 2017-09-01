"""
Download and store FastText pretrained models as files
up under this python package.
"""
import importlib
from pathlib import Path

import lmdb
import numpy as np
cimport numpy as np
import requests
from tqdm import tqdm
import numpy.linalg as la

URL_TEMPLATE = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{0}.vec"


cdef float e = np.finfo(np.float32).eps

cdef class FastTextLanguageModel:
    """
    Language model that will return word embedding vectors for word strings, with
    a unique sub-word model of ngrams allowing encoding of out of vocabulary words.

    Attributes
    ----------
    dimensions: int
        number of dimensions for each word vector
    """
    cdef object txn
    cdef np.ndarray zeros
    cdef int dimensions

    def __init__(self, language: str):
        """
        Construct a language model for a given string by:
        - opening an existing model if present
        - downloading and compiling pretrained word models otherwise

        Parameters
        ----------
        language:
        Two letter language code.
        """

        # the local in package file path for the language model
        pkg = importlib.import_module('vectoria')
        vectoria_path = Path(pkg.__file__).parent
        folder_path = vectoria_path / Path(language)
        if not folder_path.exists():
            folder_path.mkdir()
        vectors_path = folder_path / Path('{0}.vec'.format(language))
        # download if needed
        if not vectors_path.exists():
            url = URL_TEMPLATE.format(language)
            print(url)
            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)
            # Total size in bytes.
            total_size = int(r.headers.get('content-length', 0))
            with open(str(vectors_path), 'wb') as f:
                chunk = 32 * 1024
                progress = tqdm(total=total_size, unit='B', unit_scale=True)
                for data in r.iter_content(chunk):
                    if data:
                        f.write(data)
                        progress.update(len(data))
        # vectors are local by this point -- time for the database
        database_path = folder_path / Path('lmdb')
        if not database_path.exists():
            env = lmdb.open(str(database_path), map_size=12 *
                            1024 * 1024 * 1024, writemap=True)
            with env.begin(write=True) as txn:
                # now loop the whole vector file -- and encode as dense floats stored by word
                with open(str(vectors_path), 'r') as f:
                    first_line = f.readline()
                    words, dimensions = map(int, first_line.split())
                for line in tqdm(iterable=open(str(vectors_path)), total=words):
                    # first line processing, make sure we have enough segments
                    segments = line.split()
                    if len(segments) > dimensions:
                        try:
                            word = segments[0]
                            numbers = np.array(
                                list(map(np.float32, segments[1:])))
                            txn.put(word.encode('utf8'), numbers.tobytes())
                        except ValueError:
                            pass
        else:
            env = lmdb.open(str(database_path))
        self.txn = env.begin()
        with self.txn.cursor() as c:
            for key, buffer in c:
                self.dimensions = self.decode(key).shape[0]
                self.zeros = np.zeros(self.dimensions, dtype=np.float32)
                break

    cdef np.ndarray decode(self, bytes word):
        """
        Look up a vector for a single word token.
        Parameters
        ----------
        word:
            A word string, which will be encoded as a float vector.

        Returns
        -------
        A numpy array representing the word. 
        """
        buffer = self.txn.get(word)
        if buffer:
            return np.frombuffer(buffer, dtype=np.float32)
        else:
            return self.zeros

    def __getitem__(self, str word) -> np.array:
        """
        Return the dense vector encoding for the passed word

        Parameters
        ----------
        word:
            A word string, which will be encoded as a float vector.

        Returns
        -------
        A numpy array representing the word.
        """
        cdef int min_n = 3
        cdef int max_n = 6
        w_len = len(word)
        cdef np.ndarray word_vector = np.zeros(self.dimensions, dtype=np.float32)
        for n in range(min_n, max_n + 1):
            offset = 0
            ngram = word[offset:offset + n]
            word_vector = word_vector + self.decode(ngram.encode('utf8'))
            while offset + n < w_len:
                offset += 1
                ngram = word[offset:offset + n]
                word_vector = word_vector + self.decode(ngram.encode('utf8'))
            if offset == 0:   # count a short word (w_len < n) only once
                break
        #accumulate and normalize

        cdef float norm = la.norm(word_vector)
        cdef np.ndarray ret = word_vector / (norm + e)
        return ret
