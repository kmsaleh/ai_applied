import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec

# 1. DEFINE sentences  ← this was missing
#sentences = [
#    ["the", "cat", "sat", "on", "the", "mat"],
#    ["the", "dog", "sat", "on", "the", "log"],
#    ["a", "cat", "ran", "fast"],
#

# 2. Train model
#model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, seed=42)


import gensim.downloader as api

# Load GloVe model (approximately 130MB, suitable for entry-level experimentation)
# Note: At this point model is already a KeyedVectors object, no need to use .wv
model = api.load("glove-wiki-gigaword-100")

# Test the effect immediately
print(model.most_similar("cat"))
print(model.similarity("cat", "dog"))

# 3. Extract vectors
v_cat = model["cat"]
v_dog = model["dog"]cd ..

# 4. Manual cosine
cos = np.dot(v_cat, v_dog) / (norm(v_cat) * norm(v_dog))
print("Manual cosine: ", cos)

# 5. Compare with Gensim
print("Gensim similarity:", model.similarity("cat", "dog"))