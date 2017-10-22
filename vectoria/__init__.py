"""
Root module sets up all our imports.
"""
from pkg_resources import get_distribution

from .Embeddings import (CharacterTrigramEmbedding, FastTextEmbedding,
                         WordEmbedding)

__version__ = get_distribution('vectoria').version
