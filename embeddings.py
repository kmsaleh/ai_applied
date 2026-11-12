# pip install gensim
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["a", "cat", "ran", "fast"],
    # ... many more for real training
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)
print("cat vector:", model.wv["cat"][:5])
print("similar to cat:", model.wv.most_similar("cat"))