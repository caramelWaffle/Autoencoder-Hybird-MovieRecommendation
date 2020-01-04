from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from tqdm import tqdm


class similarity_helper:
    item_information_matrix = []
    similarity_matrix = []

    def __init__(self, matrix):
        self.item_information_matrix = matrix
        self.similarity_matrix = np.zeros((len(matrix), len(matrix)))

    def get_similarity_scores(self):
        print("Calcurating similarity")
        for i in tqdm(range(len(self.item_information_matrix))):
            # time.sleep()
            for j in range(len(self.item_information_matrix)):
                self.similarity_matrix[i][j] = cosine_similarity(self.item_information_matrix[i].reshape(1, -1),
                                                                 self.item_information_matrix[j].reshape(1, -1))
        return self.similarity_matrix

    def get_similarity_by_movie(self, movie_id, dict_link, ):
        index = dict_link.get(int(movie_id))
        target_vector = self.item_information_matrix[index]
        similarity_score = np.zeros(len(self.item_information_matrix))
        for i in range(len(self.item_information_matrix)):
            similarity_score[i] = cosine_similarity(target_vector.reshape(1, -1), self.item_information_matrix[i].reshape(1, -1))
        return similarity_score

    def get_movie_by_preferences(self, user_preference):
        similarity_score = np.zeros(len(self.item_information_matrix))
        for i in range(len(self.item_information_matrix)):
            similarity_score[i] = cosine_similarity(user_preference, self.item_information_matrix[i].reshape(1, -1))
        return similarity_score
