import numpy as np


class MF():

    def __init__(self, R, K, alpha, lamb, iterations, Q, batch_size, P, start=0):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - lamb (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.lamb = lamb
        self.iterations = iterations
        self.Q = Q
        self.P = P
        self.error_ = 10000
        self.batch_size = batch_size
        self.start = start

    def train(self):
        # Initialize user and item latent feature matrice
        if self.P is None:
            self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        # self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples that R[ij] > 0
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        m = len(self.samples)

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        mse = self.error_ - 1
        # i = 0
        # while(self.error_ - mse > 1e-10):
        for i in range(self.start, self.iterations):
            np.random.shuffle(self.samples)
            # self.sgd()
            self.minibatch_gradient_descent(m)
            self.error_ = mse
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

            if (i + 1) % 100 == 0:
                p_path = "checkpoint/P_" + str(self.K) + "_" + str(i + 1) + ".npy"
                q_path = "checkpoint/Q_" + str(self.K) + "_" + str(i + 1) + ".npy"

                # {self.K}_{(i + 1)}
                np.save(p_path, self.P)
                np.save(q_path, self.Q)
                print("Save P and Q step", str((i + 1)), " as /checkpoint")
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error +=  pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt((1/len(ys)) *error)

    def sgd(self):
        """
        Perform stochastic graident descent
        i = user
        j = movie
        r = rating
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.lamb * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.lamb * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.lamb * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * P_i - self.lamb * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        if isinstance(i, list):
            prediction = np.zeros(len(i))
            for iter in range(len(i)):
                prediction[iter] = self.b + self.b_u[i[iter]] + self.b_i[j[iter]] + self.P[i[iter], :].dot(
                    self.Q[j[iter], :].T)
        else:
            prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(mf):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:, np.newaxis] + mf.b_i[np.newaxis:, ] + mf.P.dot(mf.Q.T)

    def minibatch_gradient_descent(self, m):
        '''
        X    = Matrix of X without added bias units
        y    = Vector of Y
        theta=Vector of thetas np.random.randn(j,1)
        learning_rate
        iterations = no of iterations

        Returns the final theta vector and array of cost history over no of iterations
        '''
        for i in range(0, m, self.batch_size):
            samples = self.samples[i:i + self.batch_size]
            users = [samples[0] for samples in samples]  # i
            movies = [samples[1] for samples in samples]  # j
            ratings = [samples[2] for samples in samples]  # r

            prediction = self.get_rating(users, movies).astype('float128')

            # print(prediction)
            e = (ratings - prediction).astype('float128')
            print(i)
            print(e)

            # Update biases
            self.b_u[users] += self.alpha * (e - self.lamb * self.b_u[users]).astype('float128')
            self.b_u[users][np.isnan(self.b_u[users])] = 0

            self.b_i[movies] += self.alpha * (e - self.lamb * self.b_i[movies]).astype('float128')
            self.b_i[movies][np.isnan(self.b_i[movies])] = 0
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[users, :][:].astype('float128')
            P_i[np.isnan(P_i)] = 0

            # Update user and item latent feature matrices
            self.P[users, :] += self.alpha * (e.reshape((-1, 1)) * self.Q[movies, :] - self.lamb * self.P[users, :]).astype('float128')
            self.P[users, :][np.isnan(self.P[users, :])] = 0
            self.Q[movies, :] += self.alpha * (e.reshape((-1, 1)) * P_i - self.lamb * self.Q[movies, :]).astype('float128')
            self.Q[movies, :][np.isnan(self.Q[movies, :])] = 0
