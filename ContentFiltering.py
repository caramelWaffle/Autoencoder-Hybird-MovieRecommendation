import pandas as pd
import numpy as np
import collections

from EmbeddingTask import EmbeddingUtil
from CosineSimilarityTask import CosineSimilarityTask


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

try:
    item_information_matrix = np.load('data/item_information_matrix.npy')

except FileNotFoundError as e:

    print("Read item_information_matrix file")
    item_information_matrix = np.zeros((len(movieContent), len(dictionary)))
    loop = 1

    for key, value in movieContent.items():
        print("Embedding ", loop, " of ", len(movieContent))
        embeddings = EmbeddingUtil(value, model="bow", dictionary=dictionary).get_embedding()
        item_information_matrix[movie_index_link.index(key)] = embeddings
        loop = loop + 1

    np.save('data/item_information_matrix.npy', item_information_matrix)


try:
    item_similarity_score = np.load('data/item_similarity_score.npy')
except FileNotFoundError as e:
    print("Generate item_similarity_score file")
    item_similarity_score = CosineSimilarityTask(item_information_matrix).get_similarity_scores()
    np.save('data/item_similarity_score.npy', item_information_matrix)

item_similarity_df = pd.DataFrame(item_similarity_score.toarray(), index=tags.index.tolist())
print(item_similarity_df)


