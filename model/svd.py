import pandas as pd
import numpy as np
import os
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import recmetrics

from surprise.model_selection import cross_validate
from model.csv_helper import CSVHelper

os.chdir('/Users/macintoshhd/thesis_recommendation')
ratings = pd.read_csv('data/ratings.csv')
ratings.reset_index(drop=True, inplace=True)

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

# #train SVD recommender
algo = SVD()
algo.fit(trainset)

#make predictions on test set.
test = algo.test(testset)
test = pd.DataFrame(test)
test.drop("details", inplace=True, axis=1)
test.columns = ['userId', 'movieId', 'actual', 'cf_predictions']
print(test.head(10))

print("MSE : ", recmetrics.mse(test.actual, test.cf_predictions))
print("RMSE : ", recmetrics.rmse(test.actual, test.cf_predictions))
#
# full_test = algo.test(list(data.df.values))
# full_test = pd.DataFrame(full_test)
# full_test.drop("details", inplace=True, axis=1)
# full_test.columns = ['userId', 'movieId', 'actual', 'cf_predictions']
#
# print(full_test.head(10))
# print("MSE : ", recmetrics.mse(full_test.actual, full_test.cf_predictions))
# print("RMSE : ", recmetrics.rmse(full_test.actual, full_test.cf_predictions))
#
# CSVHelper().write_csv('output', 'predicted_df', full_test)
# # full_test.to_csv('output/predicted_df.csv', encoding='utf-8')

users = np.unique(ratings['userId'], axis=0)
# for user in users: