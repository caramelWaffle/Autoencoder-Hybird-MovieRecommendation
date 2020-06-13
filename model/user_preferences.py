from model.csv_helper import CSVHelper
from model.data_manager import data_manager
from model.embedding import embedding_helper
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
import collections
import pandas as pd
import numpy as np
import json


class UserPreferences:

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def get_user_preferences(self, user_id, indicate_rating_score, is_gen_csv=False):
        user_liked_movies = self.data_manager.get_ratings_from_user(user_id, indicate_rating_score)
        user_movie_df = pd.DataFrame()
        content_freq = collections.Counter()

        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        print("Extracting user liked movie's contents")
        for index, row in tqdm(user_liked_movies.iterrows(), total=user_liked_movies.shape[0]):
            user_movie_df.loc[index, "movieId"] = row['movieId']
            user_movie_df.loc[index, "title"] = self.data_manager.get_title_by_id(row['movieId'])
            user_movie_df.loc[index, "content"] = self.data_manager.list_to_string(
                self.data_manager.generate_movies_content_by_id(row['movieId']))
            user_movie_df.loc[index, "rating"] = row['rating']

            content = self.data_manager.generate_movies_content_by_id(row['movieId'])
            tokens = tokenizer.tokenize(str(content).lower().replace('sci-fi', 'scifi'))
            filtered_token = [w for w in tokens if w not in stop_words]
            cleaned_content = [stemmer.stem(lemmatizer.lemmatize(w)) for w in filtered_token]
            cleaned_content.append(
                self.data_manager.get_decades_by_title(self.data_manager.get_title_by_id(row['movieId'])))
            content_freq += collections.Counter(cleaned_content)

        if is_gen_csv:
            CSVHelper().write_csv("output", f"user_{user_id}_liked_movies", user_movie_df)

        median = np.mean(list(content_freq.values()))
        user_preferences_dic = {x: content_freq[x] for x in content_freq if content_freq[x] >= median}
        json_data = json.dumps(user_preferences_dic)
        with open('output/user-preference.json', 'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        return collections.OrderedDict(sorted(user_preferences_dic.items(), key=itemgetter(1), reverse=True))
