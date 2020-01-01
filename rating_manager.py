from data_manager import data_manager
import pandas as pd


class rating_helper:
    rating = pd.DataFrame()
    data_manager = data_manager()

    def __init__(self):
        self.rating = self.data_manager.get_ratings()

    def get_ratings_from_user(self, user_id, indicate_rating_score):
        self.data_manager.set_movie(self.rating.query(f'userId =={user_id}'))
        return self.data_manager.generate_movies_content()
