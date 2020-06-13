from model.similarity_helper import similarity_helper
import numpy as np

encoded_data = np.load("/Users/macintoshhd/thesis_recommendation/heat_map/encoded_movie_contents.npy")
encoded_data = encoded_data[:10]
cosine_similarity = similarity_helper(encoded_data)
all_movie_similarity = cosine_similarity.get_similarity_scores()
np.save("/Users/macintoshhd/thesis_recommendation/heat_map/all_encoded_movie_similarity.npy", all_movie_similarity)
