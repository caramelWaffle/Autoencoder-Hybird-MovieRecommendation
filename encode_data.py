from model.autoencoder import AutoEncoder
import numpy as np

all_data = np.load("/content/drive/My Drive/python_code/project/data/similarity_matrix.npy")
ae = AutoEncoder(all_data, validation_perc=0.1, lr=1e-3, intermediate_size=100, encoded_size=50, is_enable_bath_norm=True)
ae.train_loop(epochs=5)