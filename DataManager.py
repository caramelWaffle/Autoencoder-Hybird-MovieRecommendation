import pandas as pd
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


class DataManager:
    movies = pd.DataFrame()
    rating = pd.DataFrame()
    tags = pd.DataFrame()
    movie_content = {}
    dictionary = {}
    movie_dict_link = {}

    def __init__(self):
        self.movies = pd.read_csv("data/movies.csv", encoding='utf-8')
        self.ratings = pd.read_csv("data/ratings.csv", encoding='utf-8')
        self.tags = pd.read_csv("data/tags.csv", encoding='utf-8')

    def generate_movies_content(self):
        for index, row in self.movies.iterrows():
            self.movie_dict_link[row['movieId']] = index
            self.movie_content[row['movieId']] = row['genres'].split("|")
        for index, row in self.tags.iterrows():
            movie_content_ = self.movie_content[row['movieId']]
            movie_content_.append(row['tag'])
            self.movie_content[row['movieId']] = movie_content_
        return self.movie_content

    def generate_dictionary(self):
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))

        for index, value in self.movie_content.items():
            for item in value:
                tokens = tokenizer.tokenize(item.lower())
                filtered_token = [w for w in tokens if w not in stop_words]

                for token in filtered_token:
                    if stemmer.stem(lemmatizer.lemmatize(token)) not in self.dictionary.keys():
                        self.dictionary[stemmer.stem(lemmatizer.lemmatize(token))] = 1
                    else:
                        self.dictionary[stemmer.stem(lemmatizer.lemmatize(token))] += 1

        self.dictionary = collections.OrderedDict(sorted(self.dictionary.items()))
        return self.dictionary

    def get_movies(self):
        return self.movies

    def get_ratings(self):
        return self.ratings

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

    def get_title_by_id(self, movieId):
        return self.movies.iloc[self.movie_dict_link.get(movieId)].title

    def list_to_string(self, list_item):
        return ' '.join(map(str, list_item)).lower()
