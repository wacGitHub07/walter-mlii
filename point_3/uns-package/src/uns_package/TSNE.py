import numpy as np

class TSNE():

    def __init__(self, n_components:int=2, perplexity:float=30.0, max_iter:int=200, learning_rate:int=500)->None:
        """
        Class to t-Distributed Stochastic Neighbor Embedding implementation.
        
        PARAMS
        ----------
        max_iter : int, default 200
            Max number of iterations
        perplexity : float, default 30.0
            How to balance attention between local and global aspects of your data 
            (5 and 50) recomended values
        n_components : int, default 2
            Number of components to 
        """
        self.max_iter = max_iter
        self.perplexity = perplexity
        self.n_components = n_components
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.min_gain = 0.01
        self.lr = learning_rate
        self.tol = 1e-5
        self.perplexity_tries = 50
        self.n_samples = None


    def __l2_distance(self, X:np.ndarray) -> float:
        """
        Function to compute L2 distance

        PARAMS:
        X: ndarray
            Input data
        """
        sum_X = np.sum(X * X, axis=1)
        return (-2 * np.dot(X, X.T) + sum_X).T + sum_X


    def fit_transform(self, X:np.ndarray, y=None):
        """
        Function to compute tsne method

        PARAMS:
        X: ndarray
            Input Data
        """
        if self.n_components > X.shape[1]:
            raise Exception("Number of components must be less or equal than number of original features")

        self.n_samples = X.shape[0]

        Y = np.random.randn(self.n_samples, self.n_components)
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)

        P = self.__get_pairwise_affinities(X)

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            D = self.__l2_distance(Y)
            Q = self.__q_distribution(D)

            # Normalizer q distribution
            Q_n = Q / np.sum(Q)

            # Early exaggeration & momentum
            pmul = 4.0 if iter_num < 100 else 1.0
            momentum = 0.5 if iter_num < 20 else 0.8

            # Perform gradient step
            grads = np.zeros(Y.shape)
            for i in range(self.n_samples):
                grad = 4 * np.dot((pmul * P[i] - Q_n[i]) * Q[i], Y[i] - Y)
                grads[i] = grad

            gains = (gains + 0.2) * ((grads > 0) != (velocity > 0)) + (gains * 0.8) * ((grads > 0) == (velocity > 0))
            gains = gains.clip(min=self.min_gain)

            velocity = momentum * velocity - self.lr * (gains * grads)
            Y += velocity
            Y = Y - np.mean(Y, 0)

            error = np.sum(P * np.log(P / Q_n))
            print("Iteration %s, error %s" % (iter_num, error))
        return Y


    def __get_pairwise_affinities(self, X:np.ndarray) -> np.ndarray:
        """
        Computes pairwise affinities.
        
        PARAMS:
        X:ndarray
            Input data
        """
        affines = np.zeros((self.n_samples, self.n_samples), dtype=np.float32)
        target_entropy = np.log(self.perplexity)
        distances = self.__l2_distance(X)

        for i in range(self.n_samples):
            affines[i, :] = self.__binary_search(distances[i], target_entropy)

        # Fill diagonal with near zero value
        np.fill_diagonal(affines, 1.0e-12)

        affines = affines.clip(min=1e-100)
        affines = (affines + affines.T) / (2 * self.n_samples)
        return affines
    

    def __binary_search(self, dist:float, target_entropy:float) -> float:
        """
        Performs binary search to find suitable precision.
        
        PARAMS:
        dist: float
            L2 distance
        target_entropy: float
            Value of the entropy for compute the error
        """
        precision_min = 0
        precision_max = 1.0e15
        precision = 1.0e5

        for _ in range(self.perplexity_tries):
            denom = np.sum(np.exp(-dist[dist > 0.0] / precision))
            beta = np.exp(-dist / precision) / denom

            # Exclude zeros
            g_beta = beta[beta > 0.0]
            entropy = -np.sum(g_beta * np.log2(g_beta))

            error = entropy - target_entropy

            if error > 0:
                # Decrease precision
                precision_max = precision
                precision = (precision + precision_min) / 2.0
            else:
                # Increase precision
                precision_min = precision
                precision = (precision + precision_max) / 2.0

            if np.abs(error) < self.tol:
                break

        return beta

    def __q_distribution(self, D:np.ndarray) -> np.ndarray:
        """
        Computes Student t-distribution.
        
        PARAMS:
        D: ndarray
            Data values
        """
        Q = 1.0 / (1.0 + D)
        np.fill_diagonal(Q, 0.0)
        Q = Q.clip(min=1e-100)
        return Q