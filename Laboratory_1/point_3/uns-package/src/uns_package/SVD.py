import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


class SVD():
    """
    Class to implement the svd descomposition with two methods:
        - Power Iterate : Usefull only for square matrix
        - Power Iterate : Usefull if power iterate fails
    """

    def __init__(self, epsilon:float=1e-10, method:str='eig') -> None:
        """
        SVD class constructor:

        PARAMS:
        epsilon:float
            value to calcalte the error in the power iterations finding the eigenvectors
        metod:str
            Defining if the method is power iterate or eigenvector
        """
        self.M = None
        self.m = None
        self.n = None
        self.method = method
        self.epsilon = epsilon

        if self.method not in ('pow', 'eig'):
            raise Exception("select a valid method: pow, eig")

        self.eigenvectors_v = None
        self.eigenvectors_u = None
        self.ncols_v = None
        self.ncols_u = None
        self.eingevalues_s = None
        self.change_of_basis = None


    def fit(self, M:np.ndarray):
        """
        Function to calculate the eigenvectors, eigenvalues and
        singular values in svd descomposition

        PARAMS:
        M:ndarray
            Input Data
        """
        self.M = M
        self.m, self.n = self.M.shape
        print(f"{self.m} {self.n}")
        if self.m != self.n and self.method == 'pow':
            raise Exception("The methos pow is only for sqare matrix")

        if self.method == 'eig':
            self.__svd_eingevectors()

        elif self.method == 'pow':
            self.__svd_power_iterate(self.epsilon)

    
    def transform(self) -> tuple:
        """
        Function to build U, SIGMA, VT values in
        svd descomposition
        """
        if self.method == 'eig':
            vt = self.eigenvectors_v[:, self.ncols_v].T
            u = self.eigenvectors_u[:, self.ncols_u]
            sigma = self.eingevalues_s
        elif self.method == 'pow':
            sigma, u, vt = [np.array(x) for x in zip(*self.change_of_basis)]
            u = u.T

        return u, sigma, vt
    

    def fit_transform(self, M:np.ndarray) -> tuple:
        """
        Function to implement fit and transform at the same time

        PARAMS:
        M: ndarray
            Input Data
        """

        self.fit(M)
        return self.transform()


    def __svd_eingevectors(self) -> None:
        """
        Function to calculate eigenvectors and eigenvalues by the
        eig method
        """
        # Matrix V
        new_v = np.dot(self.M.T, self.M)
        # Matrix U 
        new_u = np.dot(self.M, self.M.T) 

        # Eingevalues and eingevectors
        eigenvalues_v, eigenvectors_v = np.linalg.eig(new_v)
        eigenvalues_u, eigenvectors_u = np.linalg.eig(new_u)
        ncols_v = np.argsort(eigenvalues_v)[::-1]
        ncols_u = np.argsort(eigenvalues_u)[::-1]

        if np.size(np.dot(self.M, self.M.T)) > np.size(np.dot(self.M.T, self.M)):
            eingevalues_s = np.sqrt(eigenvalues_v)
        else:
            eingevalues_s = np.sqrt(eigenvalues_u)
        eingevalues_s[::-1].sort()

        self.eigenvectors_v = eigenvectors_v
        self.eigenvectors_u = eigenvectors_u
        self.ncols_v = ncols_v
        self.ncols_u = ncols_u
        self.eingevalues_s = eingevalues_s


    def __random_unit_vector(self, size:int) -> list:
        """
        Function to initialize random vector in power iterate

        PARAMS:
        size: int
            Vector size
        """
        unnormalized = [normalvariate(0, 1) for _ in range(size)]
        norm = sqrt(sum(v * v for v in unnormalized))
        return [v / norm for v in unnormalized]


    def __power_iterate(self, X:np.ndarray, epsilon:float=1e-10) -> list:    
        """
        Function to recursively compute X^T X dot v to compute weights vector/eignevector 
        
        PARAMS:
        X: ndarray
            Input data
        epsilon:float
            Error value
        """
    
        n, m = X.shape
        start_v = self.__random_unit_vector(m) # start of random surf
        prev_eigenvector = None
        curr_eigenvector = start_v
        covariance_matrix = np.dot(X.T, X)
    
        ## power iterationn until converges
        it = 0        
        while True:
            it += 1
            prev_eigenvector = curr_eigenvector
            curr_eigenvector = np.dot(covariance_matrix, prev_eigenvector)
            curr_eigenvector = curr_eigenvector / norm(curr_eigenvector)
    
            if abs(np.dot(curr_eigenvector, prev_eigenvector)) > 1 - epsilon:            
                return curr_eigenvector


    def __svd_power_iterate(self, epsilon:float=1e-10) -> None:
        """
        Function to calculate eigenvectors and eigenvalues by the
        power iterate  method
        
        PARAMS:
        epsilon:float
            Error value
        """
        change_of_basis = []

        for i in range(self.n):
            data_matrix = self.M.copy()
 
            for sigma, u, v in change_of_basis[:i]:
                data_matrix -= sigma * np.outer(u, v) 
    
            v = self.__power_iterate(data_matrix, epsilon=epsilon) ## eigenvector 
            u_sigma = np.dot(self.M, v) ## 2nd step: XV = U Sigma 
            sigma = norm(u_sigma)  
            u = u_sigma / sigma
    
            change_of_basis.append((sigma, u, v))

        self.change_of_basis = change_of_basis