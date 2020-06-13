import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import defaultdict
import heapq
import math
import argparse
import pandas as pd

def load_data():
    '''
    As for bpr experiment, all ratings are removed.
    '''
    data_path = "/Users/macintoshhd/thesis_recommendation/output/processed_data.csv"
    data_df = pd.read_csv(data_path, encoding='utf-8')
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1

    for index, row in data_df.iterrows():
        u = int(row['user'])
        i = int(row['movie'])
        user_ratings[u].add(i)
        max_u_id = max(u, max_u_id)
        max_i_id = max(i, max_i_id)

    return max_u_id, max_i_id, user_ratings


user_count, item_count, user_ratings = load_data()