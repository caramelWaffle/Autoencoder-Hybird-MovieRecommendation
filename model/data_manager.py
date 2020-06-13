import pandas as pd
import collections
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer


def get_year(title):
    try:
        year = int(title.strip()[-5:-1])
    except ValueError as e:
        year = 0
    return year


def get_decades(year):
    if year == 0:
        return ''
    else:
        return f'{year}s' if int(year) % 10 == 0 else f'{int(year) - (int(year) % 10)}s'


def token_text(text):
    return TweetTokenizer().tokenize(text)

class data_manager:
    movies = pd.DataFrame()
    ratings = pd.DataFrame()
    tags = pd.DataFrame()
    movie_content = {}
    dictionary = {}
    movie_index = {}
    current_index = 0



    def __init__(self, movie_path, rating_path, tags_path):
        self.movies = pd.read_csv(movie_path, encoding='utf-8')
        self.ratings = pd.read_csv(rating_path, encoding='utf-8')
        self.tags = pd.read_csv(tags_path, encoding='utf-8')


    def generate_movies_content_by_id(self, movie_id):
        movie = self.movies.loc[self.movies['movieId'] == movie_id]
        content = movie.iloc[0]['genres'].split("|")
        content.append(get_decades(get_year(movie.iloc[0]['title'])))
        # content.extend(movie.iloc[0]['title'].split(" ")[0:-1])
        return content

    def generate_movies_content(self):

        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))

        print("\nGenerate movie contents")
        for index, row in tqdm(self.movies.iterrows(), total=self.movies.shape[0]):
            content = []
            self.movie_index[row['movieId']] = self.current_index
            content.append(row['movieId'])
            content.append(str(row['title']))
            content.extend(row['genres'].lower().split("|"))
            content.append(get_decades(get_year(row['title'])))
            self.movie_content[row['movieId']] = content
            # self.movie_content[row['movieId']] = [stemmer.stem(lemmatizer.lemmatize(str(content_))) for content_ in content]
            self.current_index += 1

        print("\nGenerate movie tags")
        for index, row in tqdm(self.tags.iterrows(), total=self.tags.shape[0]):
            movie_content_ = self.movie_content[row['movieId']]
            movie_content_.append(row['tag'])
            movie_content_ = list(dict.fromkeys(movie_content_))  # remove duplicate elements
            self.movie_content[row['movieId']] = movie_content_
            # self.movie_content[row['movieId']] = [stemmer.stem(lemmatizer.lemmatize(str(movie_content__))) for movie_content__ in movie_content_]
        return self.movie_content

    def generate_dictionary(self):
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))

        print("\nGenerating Movie Dictionary")

        for index, value in tqdm(self.movie_content.items()):
            movieId = value[0]
            value.remove(movieId)
            for item in set(value):

                tokens = tokenizer.tokenize(str(item).lower().replace('sci-fi', 'scifi'))
                filtered_token = [w for w in tokens if w not in stop_words]

                for token in filtered_token:
                    if stemmer.stem(lemmatizer.lemmatize(token)) not in self.dictionary.keys():
                        self.dictionary[stemmer.stem(lemmatizer.lemmatize(token))] = 1
                    else:
                        self.dictionary[stemmer.stem(lemmatizer.lemmatize(token))] += 1

            decade = get_decades(get_year(self.get_title_by_id(movieId)))
            if decade not in self.dictionary.keys():
                self.dictionary[decade] = 1
            else:
                self.dictionary[decade] += 1

        self.dictionary = collections.OrderedDict(sorted(self.dictionary.items()))
        return self.dictionary

    # def get_rating_matrix(self):
    #     users = self.get_user()
    #     rating_matrix = np.zeros([self.movies.shape[0], users.shape[0]])
    #     R = np.zeros([self.movies.shape[0], users.shape[0]])
    #     for i in tqdm(range(len(users))):
    #         user_ratings = data_manager.get_ratings_from_user(users[i])
    #         for index, row in user_ratings.iterrows():
    #             rating_matrix[data_manager.get_index_from_movie_id(row['movieId'])][i] = row['rating']
    #             R[data_manager.get_index_from_movie_id(row['movieId'])][i] = 1

    def get_movies(self):
        return self.movies

    def get_tags(self):
        return self.tags

    def get_movie_content(self):
        return self.generate_movies_content() if len(self.movie_content) == 0 else self.movie_content

    def get_dictionary(self):
        return self.generate_dictionary() if len(self.dictionary) == 0 else self.dictionary

    def get_movie_dict_link(self):
        if len(self.movie_dict_link) == 0:
            self.generate_movies_content()
        return self.movie_dict_link

    def list_to_string(self, list_item):
        return ' '.join(map(str, list_item)).lower()

    def set_movie(self, movies):
        self.movies = movies

    def get_title(self):
        return self.movies['title']

    def get_title_by_id(self, movieId):
        return self.movies.loc[(self.movies['movieId'] == movieId)]['title'].values[0]

    def get_content_by_id(self, movieId):
        return self.generate_movies_content_by_id(movieId)

    def get_year_by_title(self, title):
        return get_year(title)

    def get_decades_by_title(self, title):
        return get_decades(get_year(title))

    def get_index_from_movie_id(self, movie_id):
        return self.movie_index[movie_id]

    def get_movie_id_from_index(self, index):
        return list(self.movie_index.keys())[list(self.movie_index.values()).index(index)]

    def get_ratings(self):
        return self.ratings

    def get_ratings_from_user(self, user_id, indicate_rating_score=1):
        return self.ratings.loc[(self.ratings.userId == user_id) & (self.ratings['rating'] >= indicate_rating_score)]

    def get_raw_ratings_from_user(self, user_id):
        return self.ratings.loc[(self.ratings.userId == user_id)]

    def get_user(self):
        users = np.unique(self.get_ratings()['userId'], axis=0)
        return users
