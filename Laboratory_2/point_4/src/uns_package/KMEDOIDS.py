import numpy as np


class KMEDOIDS:
    """
    Class to implement kmedoids algorithm with 2 methods to calcule distance
    """

    def __init__(self, n_clusters: int=2, max_iter: int=300, check_convergence: bool=False, distance: str='euclidean',  random_state: int=1234) -> None:
        """ 
        Class constructor

        PARAMS:
            n_clusters: int 
                Number of clusters. 
            max_iter: int
                Number of times centroids will move
            check_convergence: bool
                To check if the algorithm stop or not
            distance: str
                distance method to apply ('euclidean', 'manhattan')
            random_seed: int
                model seed
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.check_convergence: bool = check_convergence
        self.medoids_cost: list = []
        self.has_converged: bool = True
        self.random_state = random_state
        if distance not in ('euclidean', 'manhattan'):
            raise ValueError('The correct distace types are: euclidean, manhattan')
        self.distance = self.euclidean_distance
        if distance == 'manhattan':
            self.distance = self.manhattan_distance
        
    def init_medoids(self, X: np.ndarray) -> None:
        """ 
        Method to init the medoids

        PARAMS
            X: ndarray
                Input data. 
        """
        self.medoids = []
        np.random.seed(self.random_state)
        #Starting medoids will be random members from data set X
        indexes = np.random.randint(0, len(X)-1,self.n_clusters)
        self.medoids = X[indexes]
        
        for i in range(0,self.n_clusters):
            self.medoids_cost.append(0)
        
    def is_converged(self, new_medoids: np.array) -> bool:
        """
        Method to check if there is convergence in the proccess

        PARAMS:
            new_medoids: ndarray 
                the recently calculated medoids to be compared with the current medoids stored in the class
        """
        return set([tuple(x) for x in self.medoids]) == set([tuple(x) for x in new_medoids])
        
    def update_medoids(self, X: np.ndarray, labels: np.ndarray)->None:
        """
        Method to update the medoids in the proccess

        PARAMS:
            X: ndarray
                Input data
            labels: 
                A list contains labels of data points
        """
        self.has_converged = True
        
        #Store data points to the current cluster they belong to
        clusters = []
        for i in range(0,self.n_clusters):
            cluster = []
            for j in range(len(X)):
                if (labels[j] == i):
                    cluster.append(X[j])
            clusters.append(cluster)
        
        #Calculate the new medoids
        new_medoids = []
        for i in range(0, self.n_clusters):
            new_medoid = self.medoids[i]
            old_medoids_cost = self.medoids_cost[i]
            for j in range(len(clusters[i])):
                
                #Cost of the current data points to be compared with the current optimal cost
                cur_medoids_cost = 0
                for dpoint_index in range(len(clusters[i])):
                    cur_medoids_cost += self.distance(clusters[i][j], clusters[i][dpoint_index])
                
                #If current cost is less than current optimal cost,
                #make the current data point new medoid of the cluster
                if cur_medoids_cost < old_medoids_cost:
                    new_medoid = clusters[i][j]
                    old_medoids_cost = cur_medoids_cost
            
            #Now we have the optimal medoid of the current cluster
            new_medoids.append(new_medoid)
        
        #If not converged yet, accept the new medoids
        if not self.is_converged(new_medoids):
            self.medoids = new_medoids
            self.has_converged = False
    
    def fit(self, X):
        """
        Method to find clusters
        
        PARAMS
            X: ndarray
              Input data. 
        """
        self.init_medoids(X)
        
        for i in range(self.max_iter):
            #Labels for this iteration
            cur_labels = []
            for medoid in range(0,self.n_clusters):
                #Dissimilarity cost of the current cluster
                self.medoids_cost[medoid] = 0
                for k in range(len(X)):
                    #Distances from a data point to each of the medoids
                    d_list = []                    
                    for j in range(0,self.n_clusters):
                        d_list.append(self.distance(self.medoids[j], X[k]))
                    #Data points' label is the medoid which has minimal distance to it
                    cur_labels.append(d_list.index(min(d_list)))
                    
                    self.medoids_cost[medoid] += min(d_list)
                                
            self.update_medoids(X, cur_labels)
            
            if self.has_converged and self.check_convergence:
                break
        
        # Format centers
        centers = []
        for center in self.medoids:
            centers.append(list(center))
        self.medoids = np.array(centers)

        return np.array(self.medoids)

        
    def predict(self,data: np.ndarray) -> np.ndarray:
        """ 
        Method to calculate the cluster of each sample in the data

        Parameters
        ----------
        data: ndarray
        input data.
        """
    
        pred = []
        for i in range(len(data)):
            #Distances from a data point to each of the medoids
            d_list = []
            for j in range(len(self.medoids)):
                d_list.append(self.distance(self.medoids[j],data[i]))
                
            pred.append(d_list.index(min(d_list)))
            
        return np.array(pred)
    
    def fit_predict(self,X: np.ndarray)->np.ndarray:
        """
        Method to execute fit and predict in the same function

        PARAMS:
            X: input data
        """
        self.fit(X)
        return self.predict(X)
    
    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Method to calcule euclidean distance between x, y

        PARAMS:
            x: ndarray
                Array values
            y: ndarray
                Array values
        """
        squared_d = 0
        for i in range(len(x)):
            squared_d += (x[i] - y[i])**2
        d = np.sqrt(squared_d)
        return d
    
    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Method to calcule manhattan distance between x, y

        PARAMS:
            x: ndarray
                Array values
            y: ndarray
                Array values
        """
        distance = 0
        for x1, x2 in zip(x, y):
            difference = x2 - x1
            absolute_difference = abs(difference)
            distance += absolute_difference
        return distance