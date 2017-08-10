"""
Transform a set of strings scikit-learn style.
"""
from vectoria import FastTextVectorizer

transformer = FastTextVectorizer(maxlen=16, language='en')

sentences = [
    'Hello World',
    'Helllllo World!',
    'I love you world',
]

#not that there is no need to 'fit', this is pretrained model
print(transformer.transform(sentences))
