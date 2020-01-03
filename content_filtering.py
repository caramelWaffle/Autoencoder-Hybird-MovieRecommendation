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

print("Read File")
data_manager = data_manager()
rating_manager = rating_helper()
csv_helper = csv_helper()
movies = data_manager.get_movies()
tags = data_manager.get_tags()

movieContent = data_manager.get_movie_content()  # key = movieId, value = content
dictionary = data_manager.get_dictionary()
movie_dict_link = data_manager.get_movie_dict_link()  # key = movieId, value = index

try:
    item_information_matrix = np.load('data/item_information_matrix.npy')
except FileNotFoundError as e:
    print("\nEmbedding content")
    item_information_matrix = np.zeros((len(movieContent), len(dictionary)))
    for movieId, value in tqdm(movieContent.items()):
        embeddings = embedding_helper(value, model="bow", dictionary=dictionary).get_embedding()
        item_information_matrix[movie_dict_link.get(movieId)] = embeddings
    np.save('data/item_information_matrix.npy', item_information_matrix)

try:
    print("\nEncoding content")
    encoded_item_matrix = np.load('data/encoded_item_information.npy')
except FileNotFoundError:
    ae = AutoEncoder(item_information_matrix, validation_perc=0.1, lr=1e-3, intermediate_size=5000, encoded_size=100)
    ae.train_loop(epochs=15)

    losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
    losses['epoch'] = (losses.index + 1) / 3

    encoded_item_matrix = ae.get_encoded_representations()
    np.save('data/encoded_item_information.npy', encoded_item_matrix)

print("\nCalculating similarity score")
mode = "user2move"

if mode == "movie2move":
    targetMovieId = 88140
    cosine_similarity = similarity_helper(encoded_item_matrix)
    movie_similarity = cosine_similarity.get_similarity_by_movie(targetMovieId, movie_dict_link)

    movie_name_column = {}  # key = movieId, value = title
    movie_content_column = {}

    for movieId, value in tqdm(movie_dict_link.items()):
        movie_name_column[movieId] = data_manager.get_title_by_id(movieId)
        movie_content_column[movieId] = data_manager.generate_movies_content_by_id(movieId)
        movie_content_column[movieId].append(data_manager.get_decades_by_title(data_manager.get_title_by_id(movieId)))
        movie_content_column[movieId] = data_manager.list_to_string(movie_content_column[movieId])

    filter_decade = True

    similar_dataframe = pd.DataFrame({'movie_id': list(movie_name_column.keys()), 'title': list(movie_name_column.values()), 'content': list(movie_content_column.values()), 'similarity_score': movie_similarity})
    similar_dataframe = similar_dataframe[similar_dataframe['title'] != data_manager.get_title_by_id(targetMovieId)]
    if filter_decade:
        temp = similar_dataframe[similar_dataframe['content'].str.contains(data_manager.get_decades_by_title(data_manager.get_title_by_id(targetMovieId)))]
        if temp.shape[0] > 100:
            similar_dataframe = temp
    similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)

    csv_helper.write_csv("output", f"{movie_name_column[targetMovieId]}_similarity_encoded", similar_dataframe)

if mode == "user2move":
    user_id = 611
    indicate_rating_score = 4
    user_perferences = user_perferences().get_user_perferences(user_id, indicate_rating_score, False)
    print(f"\nEmbedding user {user_id}'s preferences")

    user_preference_vector = np.zeros((1, len(dictionary)))
    user_preference_vector = embedding_helper(user_perferences.keys(), model="bow", dictionary=dictionary).get_embedding()
    np.save(f'data/nakhun_preference_vector.npy', user_preference_vector)

    encoded_user_preference_vector = np.load('data/encoded_nakhun_preference.npy')

    print("\nCalculating similarity score")
    cosine_similarity = similarity_helper(encoded_item_matrix)
    movie_similarity = cosine_similarity.get_movie_by_preferences(encoded_user_preference_vector)

    movie_name_column = {}  # key = movieId, value = title
    movie_content_column = {}

    for movieId, value in tqdm(movie_dict_link.items()):
        movie_name_column[movieId] = data_manager.get_title_by_id(movieId)
        movie_content_column[movieId] = data_manager.generate_movies_content_by_id(movieId)
        movie_content_column[movieId].append(data_manager.get_decades_by_title(data_manager.get_title_by_id(movieId)))
        movie_content_column[movieId] = data_manager.list_to_string(movie_content_column[movieId])

    similar_dataframe = pd.DataFrame(
        {'movie_id': list(movie_name_column.keys()), 'title': list(movie_name_column.values()),
         'content': list(movie_content_column.values()), 'similarity_score': list(movie_similarity)})

    similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)
    csv_helper.write_csv("output", f"user{user_id}_movie_similarity", similar_dataframe)

