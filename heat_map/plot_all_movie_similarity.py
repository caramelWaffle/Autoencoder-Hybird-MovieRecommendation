from model.similarity_helper import similarity_helper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


all_sim = np.load("/Users/macintoshhd/thesis_recommendation/heat_map/all_encoded_movie_similarity.npy")

all_sim = all_sim[:10, :10]

p = plt.figure(figsize=(9, 5))
sb.set(font_scale=1)
heat_map = sb.heatmap(all_sim, cmap="summer", annot=True, cbar_kws={'label': 'Similarity score'})
plt.xlabel("Movie attribute")
plt.ylabel("Movie attribute")
plt.title("encoded similarity 100k")
plt.show()

