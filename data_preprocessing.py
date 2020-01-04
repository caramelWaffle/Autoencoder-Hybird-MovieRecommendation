import pandas as pd
import numpy as np
from helper.data_manager import data_manager
from helper.embedding import embedding_helper
from helper.similarity_helper import similarity_helper
from user_preferences import user_perferences
from autoencoder import AutoEncoder
from helper.rating_manager import rating_helper
from helper.csv_helper import csv_helper
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-embedded_movie_path", type=str, default="")
parser.add_argument("-encoded_movie_path", type=str, default="")
parser.add_argument("-encoded_user_path", type=str, default="")
parser.add_argument("-target_movie_id", type=int, default="0")
parser.add_argument("-user_id", type=int, default="0")
parser.add_argument("-indicate_rating_score", type=int, default="4")
args = parser.parse_args()

user_id = args.user_id
target_movie_id = args.target_movie_id
indicate_rating_score = args.indicate_rating_score
embedded_movie_path = args.embedded_movie_path
encoded_movie_path = args.encoded_movie_path
encoded_user_path = args.encoded_user_path

print("Import dataset")
data_manager = data_manager()
rating_manager = rating_helper()
csv_helper = csv_helper()
movies = data_manager.get_movies()
tags = data_manager.get_tags()

movieContent = data_manager.get_movie_content()  # key = movieId, value = content
dictionary = data_manager.get_dictionary()
movie_dict_link = data_manager.get_movie_dict_link()  # key = movieId, value = index

print("\nEmbedding movie content")
if embedded_movie_path:
    print(os.path.abspath(embedded_movie_path))
    embedded_movie_contents = np.load(os.path.abspath(embedded_movie_path))
else:
    embedded_movie_contents = np.zeros((len(movieContent), len(dictionary)))
    for movieId, value in tqdm(movieContent.items()):
        embeddings = embedding_helper(value, model="bow", dictionary=dictionary).get_embedding()
        embedded_movie_contents[movie_dict_link.get(movieId)] = embeddings
    np.save('data/embedded_movie_content.npy', embedded_movie_contents)

print("\nEncoding content")

if encoded_movie_path:
    print(os.path.abspath(encoded_movie_path))
    encoded_movie_contents = np.load(os.path.abspath(encoded_movie_path))
else:
    ae = AutoEncoder(embedded_movie_contents, validation_perc=0.1, lr=1e-3, intermediate_size=5000, encoded_size=100)
    ae.train_loop(epochs=15)

    encoded_movie_contents = ae.get_encoded_representations()
    np.save('data/encoded_movie_contents.npy', encoded_movie_contents)

print("\nCalculating similarity score")

if target_movie_id != 0:
    cosine_similarity = similarity_helper(encoded_movie_contents)
    movie_similarity = cosine_similarity.get_similarity_by_movie(target_movie_id, movie_dict_link)

    movie_name_column = {}  # key = movieId, value = title
    movie_content_column = {}

    # TODO: Fix this
    for movieId, value in tqdm(movie_dict_link.items()):
        movie_name_column[movieId] = data_manager.get_title_by_id(movieId)
        movie_content_column[movieId] = data_manager.generate_movies_content_by_id(movieId)
        movie_content_column[movieId].append(data_manager.get_decades_by_title(data_manager.get_title_by_id(movieId)))
        movie_content_column[movieId] = data_manager.list_to_string(movie_content_column[movieId])

    filter_decade = True

    similar_dataframe = pd.DataFrame({'movie_id': list(movie_name_column.keys()), 'title': list(movie_name_column.values()), 'content': list(movie_content_column.values()), 'similarity_score': movie_similarity})
    similar_dataframe = similar_dataframe[similar_dataframe['title'] != data_manager.get_title_by_id(target_movie_id)]
    if filter_decade:
        temp = similar_dataframe[similar_dataframe['content'].str.contains(data_manager.get_decades_by_title(data_manager.get_title_by_id(target_movie_id)))]
        if temp.shape[0] > 100:
            similar_dataframe = temp
    similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)

    csv_helper.write_csv("output", f"{movie_name_column[target_movie_id]}_similarity", similar_dataframe)

if user_id != 0:
    user_perferences = user_perferences().get_user_perferences(user_id, indicate_rating_score, False)
    print(f"\nEmbedding user {user_id}'s preferences")

    if encoded_user_path:
        print(os.path.abspath(encoded_user_path))
        encoded_user_preference_vector = np.load(os.path.abspath(encoded_user_path))
    else:
        embedded_user_preference_vector = np.zeros((1, len(dictionary)))
        embedded_user_preference_vector = embedding_helper(user_perferences.keys(), model="bow",
                                                  dictionary=dictionary).get_embedding()
        np.save(f'data/user_preference_vector.npy', embedded_user_preference_vector)

        print("\nEncoding user content")
        ae = AutoEncoder(embedded_user_preference_vector, validation_perc=0.1, lr=1e-3, intermediate_size=5000, encoded_size=100, is_enable_bath_norm=False)
        ae.train_loop(epochs=15)

        encoded_user_preference_vector = ae.get_encoded_representations()
        np.save("data/encoded_user_contents", encoded_user_preference_vector)

    print("\nCalculating similarity score")
    cosine_similarity = similarity_helper(encoded_movie_contents)
    movie_similarity = cosine_similarity.get_movie_by_preferences(encoded_user_preference_vector)

    movie_name_column = {}  # key = movieId, value = title
    movie_content_column = {}

    # TODO: Fix this
    for movieId, value in tqdm(movie_dict_link.items()):
        movie_name_column[movieId] = data_manager.get_title_by_id(movieId)
        movie_content_column[movieId] = data_manager.generate_movies_content_by_id(movieId)
        movie_content_column[movieId].append(data_manager.get_decades_by_title(data_manager.get_title_by_id(movieId)))
        movie_content_column[movieId] = data_manager.list_to_string(movie_content_column[movieId])

    similar_dataframe = pd.DataFrame(
        {'movie_id': list(movie_name_column.keys()), 'title': list(movie_name_column.values()),
         'content': list(movie_content_column.values()), 'similarity_score': list(movie_similarity)})

    similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)
    csv_helper.write_csv("output", f"user_{user_id}_similarity", similar_dataframe)
    print(similar_dataframe.head(20))

