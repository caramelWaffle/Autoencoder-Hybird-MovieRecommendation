import pandas as pd
import collections
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
    rating = pd.DataFrame()
    tags = pd.DataFrame()
    movie_content = {}
    dictionary = {}
    movie_dict_link = {}

    def __init__(self):
        self.movies = pd.read_csv("data/movies.csv", encoding='utf-8')
        self.ratings = pd.read_csv("data/ratings.csv", encoding='utf-8')
        self.tags = pd.read_csv("data/tags.csv", encoding='utf-8')

    def generate_movies_content_by_id(self, movie_id):
        movie = self.movies.loc[self.movies['movieId'] == movie_id]
        content = movie.iloc[0]['genres'].split("|")
        content.extend(movie.iloc[0]['title'].split(" ")[0:-1])
        return content

    def generate_movies_content(self):
        print("\nLoading movie contents")
        for index, row in tqdm(self.movies.iterrows(), total=self.movies.shape[0]):
            content = []
            self.movie_dict_link[row['movieId']] = index
            content.append(row['movieId'])
            content.extend(row['genres'].split("|"))
            content.extend(row['title'].split(" ")[0:-1])
            self.movie_content[row['movieId']] = content

        print("\nLoading movie tags")
        for index, row in tqdm(self.tags.iterrows(), total=self.tags.shape[0]):
            movie_content_ = self.movie_content[row['movieId']]
            movie_content_.append(row['tag'])
            self.movie_content[row['movieId']] = movie_content_
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
                tokens = tokenizer.tokenize(str(item).lower())
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

    def list_to_string(self, list_item):
        return ' '.join(map(str, list_item)).lower()

    def set_movie(self, movies):
        self.movies = movies

    def get_title_by_id(self, movieId):
        return self.movies.loc[(self.movies['movieId'] == movieId)]['title'].values[0]

    def get_content_by_id(self, movieId):
        return self.generate_movies_content_by_id(self, movieId)

    def get_year_by_title(self, title):
        return get_year(title)

    def get_decades_by_title(self, title):
        return get_decades(get_year(title))



