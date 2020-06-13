import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import recmetrics
import ml_metrics as metrics
from math import log
import math

def getHitRate(predictions, actuals, k):
    if len(predictions) != len(actuals):
        raise ValueError
    hr = 0
    for i in range(len(predictions)):
        for j in range(len(actuals[i])):
            if actuals[i][j] in predictions[j][:k]:
                hr += 1
            if j == k:
                break

    return (hr / len(predictions))


def getNDCG(predictions, actuals, k):
    if len(predictions) != len(actuals):
        raise ValueError
    ndcg = 0
    for i in range(len(predictions)):
        ranked_idx = []
        for j in range(len(actuals[i])):
            if actuals[i][j] in predictions[j][:k]:
                ranked_idx.append(actuals[i].index(actuals[i][j]))
            if j == k:
                ndcg += ndcg_at_k(ranked_idx, k)
                break

    return (ndcg / len(predictions))

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg



top_N = 100
print("Top N : ", top_N)
content_full_output = os.path.join('/Users/macintoshhd/thesis_recommendation/processed', 'processed_content.csv')

with open('/Users/macintoshhd/thesis_recommendation/output/users_content_list_f.npy', 'rb') as fp:
    content_base_prediction = pickle.load(fp)

for content in tqdm(content_base_prediction):
    data_ = {'user': content['user'], 'movie': content['movie_list'][:top_N]}
    output_df = pd.DataFrame(data_)

    if not os.path.isfile(content_full_output):
        output_df.to_csv(content_full_output, index=False, encoding='utf-8-sig', header=["user", "movie"])
    else:
        output_df.to_csv(content_full_output, index=False, encoding='utf-8-sig', mode='a', header=False)


collaborative_full_output = os.path.join('/Users/macintoshhd/thesis_recommendation/processed', 'processed_collaborative.csv')
with open('/Users/macintoshhd/thesis_recommendation/output/users_collaborative_list.npy', 'rb') as fp:
    collaborative_base_prediction = pickle.load(fp)

for content in tqdm(collaborative_base_prediction):
    data_ = {'user': content['user'], 'movie': content['movie_list'][:top_N]}
    output_df = pd.DataFrame(data_)

    if not os.path.isfile(collaborative_full_output):
        output_df.to_csv(collaborative_full_output, index=False, encoding='utf-8-sig', header=["user", "movie"])
    else:
        output_df.to_csv(collaborative_full_output, index=False, encoding='utf-8-sig', mode='a', header=False)


# Combine
collaborative_df = pd.read_csv(collaborative_full_output, encoding='utf-8')
content_df = pd.read_csv(content_full_output, encoding='utf-8')

bigdata = pd.concat([collaborative_df, content_df], ignore_index=True, sort=False)
bigdata = bigdata.drop_duplicates(subset=['user', 'movie'], keep=False)


bigdata.to_csv(f'processed/processed_data_{top_N}.csv', index=False, encoding='utf-8-sig',
                 header=["user", "movie"])


with open('/Users/macintoshhd/thesis_recommendation/output/user_actual_list.npy', 'rb') as fp:
    user_actual_list = pickle.load(fp)

# with open('/Users/macintoshhd/Downloads/users_content_list_3.npy', 'rb') as fp:
#     content_list = pickle.load(fp)

collaborative_base_prediction_m = [list_m['movie_list'] for list_m in collaborative_base_prediction]
content_base_prediction_m = [list_m['movie_list'] for list_m in content_base_prediction]

collaborative_map = metrics.mapk(user_actual_list, collaborative_base_prediction_m, top_N)
content_map = metrics.mapk(user_actual_list, content_base_prediction_m, top_N)


hr_k = 10
content_hr = getHitRate(content_base_prediction_m, user_actual_list, hr_k)
colla_hr = getHitRate(collaborative_base_prediction_m, user_actual_list, hr_k)

ndcg_k = 10
content_ndcg = getNDCG(content_base_prediction_m, user_actual_list, ndcg_k)
colla_ndcg = getNDCG(collaborative_base_prediction_m, user_actual_list, ndcg_k)

print(f"collaborative MAPK@{top_N}", collaborative_map)
print(f"content base MAPK@{top_N} ", content_map)

print("===========================================")

print(f'collaborative hr@{hr_k}', colla_hr)
print(f'content base hr@{hr_k}', content_hr)

print("===========================================")

print(f'collaborative ndcg@{hr_k}', colla_ndcg)
print(f'content base ndcg@{hr_k}', content_ndcg)