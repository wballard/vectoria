"""
Python script to fetch a pretrained vector language,
and then show vector encodings of words and constituent ngrams.
"""
import vectoria

m = vectoria.load_model('./test/test.bin')
print(m['import'])

en = vectoria.load_fasttext_model('en')
print(en['in'])
print(en['hel'])
print(en['paper'])
print(en['record'])