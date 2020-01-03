from helper.data_manager import data_manager
import pandas as pd


class rating_helper:
    rating = pd.DataFrame()
    data_manager = data_manager()

    def __init__(self):
        self.rating = self.data_manager.get_ratings()

    def get_ratings_from_user(self, user_id, indicate_rating_score=1):
        return self.rating.loc[(self.rating['userId'] >= user_id) & (self.rating['rating'] >= indicate_rating_score)]
