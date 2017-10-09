"""
Download and store pretrained models that can be used to create embedding
layers.
"""
import importlib
from pathlib import Path

from . import Sequencers

import numpy as np
import requests
from tqdm import tqdm
import numpy.linalg as la

URL_TEMPLATE = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{0}.vec"


epsilon = np.finfo(np.float32).eps

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

class CharacterTrigramFastText:
    """
    Language model base that will download and compile pretrained FastText vectors
    for a given language.

    Attributes
    ----------
    embeddings
        A two dimensional numpy array [term id, vector dimension] storing floating points.
        This is a memory mapped array to save some I/O.
    """

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
        vectors_path = download_path('fasttext', language)
        final_path = vectors_path.with_suffix('.numpy')
        # download if needed
        if not vectors_path.exists():
            url = URL_TEMPLATE.format(language)
            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)
            # Total size in bytes.
            total_size = int(r.headers.get('content-length', 0))
            with open(vectors_path.with_suffix('.tmp'), 'wb') as f:
                chunk = 32 * 1024
                progress = tqdm(total=total_size, unit='B', unit_scale=True)
                for data in r.iter_content(chunk):
                    if data:
                        f.write(data)
                        progress.update(len(data))
            vectors_path.with_suffix('.tmp').rename(vectors_path)
        # compile if needed
        if not final_path.exists():
            with open(vectors_path, 'r') as f:
                first_line = f.readline()
                words, dimensions = map(int, first_line.split())
                embeddings = np.memmap(final_path.with_suffix('.tmp'), dtype='float32', mode='w+', shape=(words,dimensions))
            sequencer = Sequencers.CharacterTrigramSequencer()
            for line in tqdm(iterable=open(str(vectors_path)), total=words):
                # how big is this thing?
                segments = line.split()
                if len(segments) > dimensions:
                    word = sequencer(segments[0])[0]
                    numbers = np.array(list(map(np.float32, segments[1:])))
                    embeddings[word] = numbers
            del embeddings
            final_path.with_suffix('.tmp').rename(final_path)
        # and -- actually open
        with open(vectors_path, 'r') as f:
            first_line = f.readline()
            words, dimensions = map(int, first_line.split())
            self.embeddings = np.memmap(final_path, dtype='float32', mode='w+', shape=(words,dimensions))

