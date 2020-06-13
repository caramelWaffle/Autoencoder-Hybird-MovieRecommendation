from model.data_manager import data_manager
from tqdm import tqdm
import pandas as pd
import numpy as np

import pickle
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import recmetrics
import collections
from operator import itemgetter

data_manager = data_manager("data/movies.csv", "data/ratings.csv", "data/tags.csv")
rating = data_manager.get_ratings()
movies = data_manager.get_movies()
movie_content = data_manager.get_movie_content()

X = np.load("/Users/macintoshhd/thesis_recommendation/data/encoded_movie_contents.npy",).astype('float128')
users = np.unique(rating['userId'], axis=0)

try:
    rating_matrix = np.load('data/rating_matrix.npy')
except FileNotFoundError:
    rating_matrix = np.zeros([movies.shape[0], users.shape[0]])
    R = np.zeros([movies.shape[0], users.shape[0]])

    for i in tqdm(range(len(users))):
        user_ratings = data_manager.get_ratings_from_user(users[i])
        for index, row in user_ratings.iterrows():
            rating_matrix[data_manager.get_index_from_movie_id(row['movieId'])][i] = row['rating']
            R[data_manager.get_index_from_movie_id(row['movieId'])][i] = 1

    np.save('data/rating_matrix.npy', rating_matrix)

movie_title = data_manager.get_title()

user_actual_rating = rating.copy()
del user_actual_rating['timestamp']

user_actual_list = []
user_predicted_list = []
user_predicted_list_m = []
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=1)

# #train SVD recommender
algo = SVD()
algo.fit(trainset)

#make predictions on test set.
test = algo.test(testset)
test = pd.DataFrame(test)
test.drop("details", inplace=True, axis=1)
test.columns = ['userId', 'movieId', 'actual', 'cf_predictions']

print("MSE : ", recmetrics.mse(test.actual, test.cf_predictions))
print("RMSE : ", recmetrics.rmse(test.actual, test.cf_predictions))
#
# predictions_data = []
# mean_ratings = rating.groupby(['movieId']).mean()['rating']
#
# movieIds = list(movies['movieId'])
#
# for user in tqdm(users):
#     user_movie_list = list(user_actual_rating.sort_values('rating', ascending=False).query(f'userId == {user}')['movieId'])
#     user_predict_movie_list = []
#     predicted_cf = {}
#     for movie in movieIds:
#         predicted = algo.predict(user, movie)
#         predicted_cf.update({predicted[1]: predicted[3]})
#
#     user_actual_list.append(user_movie_list)
#     user_predicted_list_m.append(list(collections.OrderedDict(sorted(predicted_cf.items(), key=itemgetter(1), reverse=True)).keys()))
#
#     user_predicted_list.append({'user': user, 'movie_list': list(collections.OrderedDict(sorted(predicted_cf.items(), key=itemgetter(1), reverse=True)).keys())})
#
#
# print("Save user_actual_list")
# with open('output/user_actual_list.npy', 'wb') as fp:
#     pickle.dump(user_actual_list, fp)
#
# print("Save users_collaborative_list")
# with open('output/users_collaborative_list.npy', 'wb') as fp:
#     pickle.dump(user_predicted_list, fp)

#
# print("=====================================")
# print("number of user ", len(users))
# print("number of movie", len(movies))
# print("collaborative MAPK ", cf_map)
# print("content base MAPK ", content_map)
#
# print("Process data")
# for predicted_list in tqdm(user_predicted_list):
#
#
#     data_ = {'user': predicted_list['user'], 'movie': predicted_list['movie_list'][:300]}
#     output_df = pd.DataFrame(data_)
#
#     if not os.path.isfile(colla_full_output):
#         output_df.to_csv(colla_full_output, index=False, encoding='utf-8-sig', header=["user", "movie"])
#     else:
#         output_df.to_csv(colla_full_output, index=False, encoding='utf-8-sig', mode='a', header=False)


#
# print("Generate Predictions")
# for user in tqdm(users):
#     for movie in movieIds:
#         # for predictions
#         if not rating.query(f'userId == {user} and movieId == {movie}').empty:
#             actual_rating = float(rating.query(f'userId == {user} and movieId == {movie}')['rating'])
#         else:
#             try:
#                 actual_rating = float(mean_ratings[movie])
#             except KeyError:
#                 actual_rating = float(algo.default_prediction())
#
#         data = {'userId': user, 'movieId': movie, 'actual_rating': actual_rating,
#                 'collaborative_rating': algo.predict(user, movie).est}
#         predictions_data.append(data)
#         predictions_df = pd.DataFrame(predictions_data)
#
#         csv_name = os.path.join('output', "predictions_dataframe.csv")
#         if not os.path.isfile(csv_name):
#             predictions_df.to_csv(csv_name, index=False, encoding='utf-8-sig',
#                              header=["userId", "movieId", "actual_rating", "collaborative_rating"])
#         else:
#             predictions_df.to_csv(csv_name, index=False, encoding='utf-8-sig', mode='a', header=False)

# similar_dataframe = pd.DataFrame({'id': list(movie_content.keys()), 'title': movie_title, 'content': list(movie_content.values()), 'predicted_rating': list(user_predicting_rating)})
# similar_dataframe = similar_dataframe.sort_values('predicted_rating', ascending=False)
# # csv_helper.write_csv("output", f"user_{user_id}_similarity", similar_dataframe)
#
# for index, row in islice(similar_dataframe.iterrows(), similar_dataframe.shape[0]):
#     # if '2010s' in row["content"]:
#     print(f'{row["id"]} {row["title"]} {row["content"]} {row["predicted_rating"]}')
#
#

