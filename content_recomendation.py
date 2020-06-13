from model.data_manager import data_manager
from model.embedding import embedding_helper
from model.similarity_helper import similarity_helper
from model.user_preferences import UserPreferences
from model.autoencoder import AutoEncoder
from model.csv_helper import CSVHelper

from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from itertools import islice
import os
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("-embedded_movie_path", type=str, default="")
parser.add_argument("-encoded_movie_path", type=str, default="")
parser.add_argument("-encoded_user_path", type=str, default="")
parser.add_argument("-target_movie_id", type=int, default="0")
parser.add_argument("-indicate_rating_score", type=int, default="4")
parser.add_argument("-user_id", type=int, default="0")
parser.add_argument("-movie_path", type=str, default="")
parser.add_argument("-rating_path", type=str, default="")
parser.add_argument("-tags_path", type=str, default="")
args = parser.parse_args()

user_id = args.user_id
movie_path = args.movie_path
rating_path = args.rating_path
tags_path = args.tags_path
target_movie_id = args.target_movie_id
indicate_rating_score = args.indicate_rating_score
embedded_movie_path = args.embedded_movie_path
encoded_movie_path = args.encoded_movie_path
encoded_user_path = args.encoded_user_path

data_manager = data_manager("data/movies.csv", "data/ratings.csv", "data/tags.csv")
csv_helper = CSVHelper()

movies = data_manager.get_movies()
tags = data_manager.get_tags()
movie_content = data_manager.get_movie_content()  # key = movieId, value = content
movie_title = data_manager.get_title()
dictionary = data_manager.get_dictionary()


print("\nEmbedding movie content")
if embedded_movie_path:
    print(embedded_movie_path)
    embedded_movie_contents = np.load(embedded_movie_path)
else:
    embedded_movie_contents = np.zeros((len(movie_content), len(dictionary)))
    for movie_id, value in tqdm(movie_content.items()):
        embeddings = embedding_helper(value, model="bow", dictionary=dictionary).get_embedding()
        embedded_movie_contents[data_manager.get_index_from_movie_id(movie_id)] = embeddings  # array start at 0
    np.save('data/embedded_movie_content.npy', embedded_movie_contents)

print("\nEncoding content")

if encoded_movie_path:
    print(encoded_movie_path)
    encoded_movie_contents = np.load(encoded_movie_path)
else:
    print(embedded_movie_contents.shape)
    intermediate_size = int(embedded_movie_contents.shape[1])
    encoded_size = int(intermediate_size)
    ae = AutoEncoder(embedded_movie_contents, validation_perc=0.2, lr=1e-3, intermediate_size=5000, encoded_size=100,
                     is_enable_bath_norm=True)
    ae.train_loop(epochs=60)
    encoded_movie_contents = ae.get_encoded_representations()
    ae.save_encoder()
    ae.save_decoder()
    print(encoded_movie_contents.shape)
    np.save('data/encoded_movie_contents.npy', encoded_movie_contents)

    losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
    losses['epoch'] = (losses.index + 1)

    fig, ax = plt.subplots()
    ax.plot(losses['epoch'], losses['train_loss'], label='train_loss')
    ax.plot(losses['epoch'], losses['validation_loss'], label='validation_loss')
    ax.set_ylabel('MSE loss')
    ax.set_xlabel('epoch')
    ax.set_title('autoencoder loss over time')
    ax.legend()
    fig.savefig('autoencoder.png')

user_preferences = UserPreferences(data_manager).get_user_preferences(user_id, indicate_rating_score, False)

if encoded_user_path:
    print(encoded_user_path)
    encoded_user_preference_vector = np.load(encoded_user_path)
else:
    # embedded_user_preference_vector = np.zeros((1, len(dictionary)))
    embedded_user_preference_vector = embedding_helper(user_preferences.keys(), model="bow",
                                                       dictionary=dictionary).get_embedding()
    np.save(f'data/user_preference_vector.npy', embedded_user_preference_vector)

    print("\nEncoding user content")
    ae = AutoEncoder(embedded_user_preference_vector, validation_perc=0.2, lr=1e-3, intermediate_size=5000,
                     encoded_size=100, is_enable_bath_norm=False)
    ae.load_encoder()
    ae.load_decoder()

    encoded_user_preference_vector = ae.get_encoded_representations()
    np.save("data/encoded_user_contents", encoded_user_preference_vector)

print("\nCalculating similarity score")
cosine_similarity = similarity_helper(encoded_movie_contents)
movie_similarity = cosine_similarity.get_movie_by_preferences(encoded_user_preference_vector)

similar_dataframe = pd.DataFrame(
    {'id': list(movie_content.keys()), 'title': movie_title, 'content': list(movie_content.values()),
     'similarity_score': list(movie_similarity)})
similar_dataframe = similar_dataframe.sort_values('similarity_score', ascending=False)
csv_helper.write_csv("output", f"user_{user_id}_similarity", similar_dataframe)

for index, row in islice(similar_dataframe.iterrows(), 100):
    print(f'{index} {row["id"]} {row["title"]} {row["content"]} {row["similarity_score"]}')

print(encoded_movie_contents.shape)