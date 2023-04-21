from math import sqrt, floor
import numpy as np


class KMEANS():
    """
    Class to implement kmeans algorithm with 3 methods to initialize vectors
    """

    def __init__(self, n_clusters: int, init: str='random', max_iters: int=1000, tol: float=1e-5, random_state = 7):
        """
        class constructor

        PARAMS:
            n_clusters: int
                Number of groups to clustering data
            init: str
                Vector inicialization method
            max_iters: int
                Maximum number of iteration for calculate the clusters
            tol: float
                Error tolerance to calcule the convergence
            random_state: int
                Model seed
        """
        self.n_clusters: int = n_clusters
        self.max_iters: int = max_iters
        self.tol: float = tol
        self.centroids: np.array = None
        self.labels: np.array = None
        if init not in ('random', '++', 'naive'):
            raise Exception("Incorrect initialize method. valid methods: random, ++, naive")
        self.init: str = init
        self.random_state:int = random_state

    def __calc_distances(self, X: np.array) -> np.array:
        """
        Method to calcule the distance between data and centroids
        
        PARAMS:
            X: ndarray
                input data
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
        
    def fit(self, X: np.array) -> np.array:
        """
        Method to calcule the centroids of the clusters

        PARAMS:
            X: ndarray
                Input data
        """
        n_features = X.shape[1]
        
        # Initialize centroids
        if self.init == 'random':
            self.centroids = self.random_init(X, n_clusters=self.n_clusters, random_state=self.random_state)
        elif  self.init == '++':
            self.centroids = self.plus_plus(X, n_clusters=self.n_clusters, random_state=self.random_state)
        else:
            self.centroids = self.naive_sharding(X, n_clusters=self.n_clusters)


        for i in range(self.max_iters):
            # Assign each X point to the nearest centroid
            distances = self.__calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, data: np.array) -> np.array:
        """
        Method to calculate the cluster of each sample in the data

        PARAMS:
            data: ndarray
                input data
        """
        distances = self.__calc_distances(data)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.array) -> np.array:
        """
        Method to execute fit and predict in the same function

        PARAMS:
            X: input data
        """
        self.fit(X)
        return self.predict(X)
    
    @staticmethod
    def random_init(X: np.ndarray, n_clusters: int, random_state: int=42) -> np.ndarray:
        """
        Method to create random cluster centroids.
        
        PARAMS:
        X : numpy array
            The dataset to be used for centroid initialization.
        n_clusters : int
            The desired number of clusters for which centroids are required.
        random_state: int
            model seed
        """

        np.random.seed(random_state)
        centroids = []
        m = np.shape(X)[0]

        for _ in range(n_clusters):
            r = np.random.randint(0, m-1)
            centroids.append(X[r])

        return np.array(centroids)
    
    @staticmethod
    def plus_plus(X: np.ndarray, n_clusters: int, random_state: int=42) -> np.array:
        """
        Method to create cluster centroids using the k-means++ algorithm.
        
        PARAMS:
        X : numpy array
            The dataset to be used for centroid initialization.
        n_clusters : int
            The desired number of clusters for which centroids are required.
        Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        """

        np.random.seed(random_state)
        centroids = [X[0]]

        for _ in range(1, n_clusters):
            dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in X])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            
            centroids.append(X[i])

        return np.array(centroids)
    
    @staticmethod
    def naive_sharding(X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Method to create cluster centroids using deterministic naive sharding algorithm.
        
        PARAMS:
        X : numpy array
            The dataset to be used for centroid initialization.
        n_clusters : int
            The desired number of clusters for which centroids are required.
        """

        def _get_mean(sums, step):
            """Vectorizable ufunc for getting means of summed shard columns."""
            return sums/step

        n = np.shape(X)[1]
        m = np.shape(X)[0]
        centroids = np.zeros((n_clusters, n))

        composite = np.mat(np.sum(X, axis=1))
        X = np.append(composite.T, X, axis=1)
        X.sort(axis=0)

        step = floor(m/n_clusters)
        vfunc = np.vectorize(_get_mean)

        for j in range(n_clusters):
            if j == n_clusters-1:
                centroids[j:] = vfunc(np.sum(X[j*step:,1:], axis=0), step)
            else:
                centroids[j:] = vfunc(np.sum(X[j*step:(j+1)*step,1:], axis=0), step)

        return centroids