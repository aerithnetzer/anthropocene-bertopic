from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.keyedvectors import KeyedVectors
from tensorboard.plugins.projector import visualize_embeddings
import tensorflow as tf
import os

documents_by_decade = {}

# Read corpus and dates
with (
    open("./corpus-6.txt", "r") as corpus_file,
    open("./categories-6.txt", "r") as dates_file,
):
    for line, date in zip(corpus_file, dates_file):
        year = date.strip().split("-")[0]  # Extract the year from the date
        decade = (int(year) // 10) * 10  # Calculate the decade
        if decade not in documents_by_decade:
            documents_by_decade[decade] = []
        documents_by_decade[decade].append(line.split())

# Train and save a Word2Vec model for each decade
for decade, documents in documents_by_decade.items():
    model = Word2Vec(
        sentences=documents, vector_size=100, window=5, min_count=1, workers=4
    )
    model.save(f"word2vec_{decade}.model")
