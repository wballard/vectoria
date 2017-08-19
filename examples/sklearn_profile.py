"""
Transform a set of strings scikit-learn style.
"""
from vectoria import FastTextVectorizer

transformer = FastTextVectorizer(maxlen=16, language='en')

sentences = [
    "Born in Portsmouth, Dickens left school to work in a factory when his father was incarcerated in a debtors' prison. Despite his lack of formal education, he edited a weekly journal for 20 years, wrote 15 novels, five novellas, hundreds of short stories and non-fiction articles, lectured and performed extensively, was an indefatigable letter writer, and campaigned vigorously for children's rights, education, and other social reforms."
]

for i in range(0, 10000):
    transformer.transform(sentences)
