import pandas as pd
import numpy as np
from data_manager import data_manager
from embedding import embedding_helper
from similarity_helper import similarity_helper
from autoencoder import AutoEncoder
from rating_manager import rating_helper

print("Read File")
data_manager = data_manager()
rating_manager = rating_helper()
movies = data_manager.get_movies()
tags = data_manager.get_tags()

movieContent = data_manager.get_movie_content()  # key = movieId, value = content
dictionary = data_manager.get_dictionary()
movie_dict_link = data_manager.get_movie_dict_link()  # key = movieId, value = index

# userPreference = rating_manager.get_ratings_from_user(user_id=1, indicate_rating_score=4)

try:
    print("Embedding content")
    item_information_matrix = np.load('data/item_information_matrix.npy')
except FileNotFoundError as e:
    print("Read item_information_matrix file")
    item_information_matrix = np.zeros((len(movieContent), len(dictionary)))
    loop = 1

    for movieId, value in movieContent.items():
        print("Embedding ", loop, " of ", len(movieContent))
        embeddings = embedding_helper(value, model="bow", dictionary=dictionary).get_embedding()
        item_information_matrix[movie_dict_link.get(movieId)] = embeddings
        loop = loop + 1
    np.save('data/item_information_matrix.npy', item_information_matrix)

try:
    print("Encoding content")
    encoded_item_matrix = np.load('data/encoded_item_information.npy')
except FileNotFoundError:
    ae = AutoEncoder(item_information_matrix, validation_perc=0.1, lr=1e-3, intermediate_size=5000, encoded_size=100)
    ae.train_loop(epochs=15)

    losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
    losses['epoch'] = (losses.index + 1) / 3

    encoded_item_matrix = ae.get_encoded_representations()
    np.save('data/encoded_item_information.npy', encoded_item_matrix)

print("Calculating similarity score")
targetMovieId = 163134
cosine_similarity = similarity_helper(encoded_item_matrix)
movie_similarity = cosine_similarity.get_similarity_by_movie(targetMovieId, movie_dict_link)

movie_name_column = {}  # key = movieId, value = title
movie_content_column = {}
for movieId, value in movie_dict_link.items():
    movie_name_column[movieId] = data_manager.get_title_by_id(movieId)
    movie_content_column[movieId] = data_manager.list_to_string(movieContent.get(movieId))

filter_decade = True

similar_dataframe = pd.DataFrame({'movie_id': list(movie_name_column.keys()), 'title': list(movie_name_column.values()), 'content': list(movie_content_column.values()), 'similarity_score': movie_similarity})
similar_dataframe = similar_dataframe[similar_dataframe['title'] != data_manager.get_title_by_id(targetMovieId)]
if filter_decade:
    temp = similar_dataframe[similar_dataframe['content'].str.contains(data_manager.get_decades_by_title(data_manager.get_title_by_id(targetMovieId)))]
    if temp.shape[0] > 100:
        similar_dataframe = temp
similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)
similar_dataframe.to_csv(f'output/{movie_name_column[targetMovieId]}_similarity_encoded', index=False)