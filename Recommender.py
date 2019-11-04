import pandas as pd
import numpy as np
import collections
from sklearn.metrics.pairwise import cosine_similarity

from EmbeddingTask import EmbeddingUtil


def list_to_string(list_item):
    return ' '.join(map(str, list_item)).lower()


movies = pd.read_csv("data/movies.csv", encoding='utf-8')
ratings = pd.read_csv("data/ratings.csv", encoding='utf-8')
tags = pd.read_csv("data/tags.csv", encoding='utf-8')

movieContent = {}
for index, row in movies.iterrows():
    movieContent[row['movieId']] = row['genres'].split("|")

for index, row in tags.iterrows():
    movieContent_ = movieContent[row['movieId']]
    movieContent_.append(row['tag'])
    movieContent[row['movieId']] = movieContent_

dictionary = {}
movie_index_link = list()

for key, value in movieContent.items():
    movie_index_link.append(key)
    for item in value:
        if item not in dictionary.keys():
            dictionary[item] = 1
        else:
            dictionary[item] += 1

dictionary = collections.OrderedDict(sorted(dictionary.items()))
item_information_matrix = np.zeros((len(movieContent), len(dictionary)))

for key, value in movieContent.items():
    # embeddings = EmbeddingUtil(list_to_string(value), model="bert", dictionary=dictionary).getEmbedding()
    embeddings = EmbeddingUtil(value, model="bow", dictionary=dictionary).getEmbedding()
    item_information_matrix[movie_index_link.index(key)] = embeddings
# cosine_similarity(movie_vector[1].reshape(1, -1), movie_vector[1].reshape(1, -1))
