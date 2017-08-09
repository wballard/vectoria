"""
Download and store FastText pretrained models as Python 'packages'
"""
import os
import subprocess
import importlib
from pathlib import Path
from .fasttext import load_model, FastTextModelWrapper
from tqdm import tqdm
import requests
from zipfile import ZipFile


URL_TEMPLATE = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{0}.zip"
URL_EN = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip"


def where_to_store(language: str) -> Path:
    """
    Compute a Path for a language to store up under the python modules.

    Parameters
    ----------
    language:
      Two letter language code.

    Returns
    -------
    A Path with to create a folder for the language model.

    """
    pkg = importlib.import_module('vectoria')
    vectoria_path = Path(pkg.__file__).parent
    return vectoria_path / Path(language)


def download_a_language(language: str):
    """
    Download, store, unzip, and place a specific language model.
    """
    if language == 'en':
        url = URL_EN
    else:
        url = URL_TEMPLATE.format(language)
    # Streaming, so we can iterate over the response.
    print(url)
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    language_folder = where_to_store(language)
    if not language_folder.exists():
        language_folder.mkdir()
    zipfilepath = language_folder / Path('model.zip')
    print(zipfilepath)
    with open(str(zipfilepath), 'wb') as f:
        chunk = 32 * 1024
        progress = tqdm(total=total_size, unit='B', unit_scale=True)
        for data in r.iter_content(chunk):
            if data:
                f.write(data)
                progress.update(len(data))
    with ZipFile(str(zipfilepath)) as z:
        z.extractall(str(language_folder))


def load_fasttext_model(language: str) -> FastTextModelWrapper:
    """
    If we have it, use it -- otherwise download, unpack and use it.
    """
    language = language.lower()
    language_folder = where_to_store(language)
    if language is 'en':
        language_file = language_folder / Path('wiki.{0}.bin'.format(language))
    else:
        language_file = language_folder / Path('wiki-news-300d-1M-subword.vec')
    print(language_file)
    if not language_file.exists():
        download_a_language(language)
    return load_model(str(language_file))
