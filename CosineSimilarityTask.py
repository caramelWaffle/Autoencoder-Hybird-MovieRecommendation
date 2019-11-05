from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from tqdm import tqdm


class CosineSimilarityTask:
    item_information_matrix = []
    similarity_matrix = []

    def __init__(self, matrix):
        self.item_information_matrix = matrix
        self.similarity_matrix = np.zeros((len(matrix), len(matrix)))

    def get_similarity_scores(self):
        for i in tqdm(range(len(self.item_information_matrix))):
            time.sleep(3)
            for j in range(len(self.item_information_matrix)):
                self.similarity_matrix[i][j] = cosine_similarity(self.item_information_matrix[i].reshape(1, -1),
                                                                 self.item_information_matrix[j].reshape(1, -1))

