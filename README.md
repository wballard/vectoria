# Vectoria
A Word Vector Encoder, used to turn word strings into dense numerical embeddings for
machine learning models.

## Words and Character Trigrams
Both word and chararacter trigram parsings are supported, though readily available
GLOVE word vectors do not provide for trigram parsing.

## Model Download
The various embedding classes will download and compile dense numpy arrays
of word vectors. Allow for 4G of space for each language model. The model
files will be cached as additional files within the downloaded and installed python
module.

## [FastText](https://github.com/facebookresearch/fastText)
FastText is a unique word encoding model that combines full words and
character ngrams, allowing encodings of unknown words to be estimated by their
constituent characters.

And, there are [pretrained vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) 
available for many languages, which means you can get started quickly. This library
will download and unpack those pretrained models for you.


Take a look at the ```examples``` folder.