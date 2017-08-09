"""
Python script to fetch a pretrained vector language,
and then show vector encodings of words and constituent ngrams.
"""
import vectoria

#this will download the model if not already present
#it is a 1G file -- make sure you have space!
en = vectoria.FastTextLanguageModel('en')
print(en['hello'])
print(en['hel'])
print(en['ell'])
print(en['llo'])
#and the power of the subword ngram model for 'words' not in vocabulary, 
print(en['hello_world'])