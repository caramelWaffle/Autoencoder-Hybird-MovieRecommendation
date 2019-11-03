import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from EmbeddingUtil import EmbeddingUtil


def list_to_string(list_item):
    return ' '.join(map(str, list_item)).lower()


movies = pd.read_csv("data/movies.csv", encoding='utf-8')
print(movies.head(10))
ratings = pd.read_csv("data/ratings.csv", encoding='utf-8')
print(ratings.head(10))
tags = pd.read_csv("data/tags.csv", encoding='utf-8')
print(tags.head(10))

movieContent = {}
for index, row in movies.iterrows():
    movieContent[row['movieId']] = row['genres'].split("|")

for index, row in tags.iterrows():
    movieContent_ = movieContent[row['movieId']]
    movieContent_.append(row['tag'])
    movieContent[row['movieId']] = movieContent_

print(movieContent)


movie_content_vec = {}

for key, value in movieContent.items():
    sentence = list_to_string(value)
    embeddings = EmbeddingUtil(sentence, model="bert").getEmbedding()
    movie_content_vec[key] = embeddings
    print(embeddings)


# cosine_similarity(movie_content_vec[1].reshape(1, -1), movie_content_vec[1].reshape(1, -1))
# print(cosine_similarity)