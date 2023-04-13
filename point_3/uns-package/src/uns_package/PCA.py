import numpy as np

class PCA():
    """
    Class to implement pca algoritm using
    eigenvalues and eigenvectors
    """

    def __init__(self, n_components:int) -> None:
        """
        PCA class contructor

        PARAMS:
        ncomponents: int
            Number of principal componets to calculate
        """
        self.n_components = n_components
        self.components = None
        self.sorted_eigenvectors = None
        self.cumulative_variance = None
        self.explained_variance = None


    def fit(self, X:np.ndarray) -> None:
        """
        Function to centralize the data input, 
        calculate the eigenvalues and eigenvectors,
        covariance matrix 

        PARAMS:
        X: ndarray
            Input Data
        """
        if self.n_components > X.shape[1]:
            raise Exception("Number of components must be less or equal than number of original features")
        
        centralized_data = X - X.mean(axis = 0)
        covariance_matrix = np.cov(centralized_data, rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
        order_of_importance = np.argsort(eigenvalues)[::-1] 

        # utilize the sort order to sort eigenvalues and eigenvectors
        sorted_eigenvalues = eigenvalues[order_of_importance]
        self.sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns

        # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
        self.explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        self.cumulative_variance = [np.sum(self.explained_variance [:i+1]) for i in range(len(self.explained_variance))]

        
    def transform(self, X:np.ndarray) -> np.ndarray:
        """
        Function to transform the original data in its principal components

        PARAMS:
        X: ndarray
            Input Data
        """

        centralized_data = X - X.mean(axis = 0)
        # transform the original data
        reduced_data = np.matmul(centralized_data, self.sorted_eigenvectors[:,:self.n_components]) 
        return reduced_data

    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        """
        Function that apply fit and transform over the input data
        
        X: ndarray
            Input Data
        """
        self.fit(X)
        return self.transform()