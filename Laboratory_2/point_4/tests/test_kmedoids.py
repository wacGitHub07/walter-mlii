from math import sqrt, floor
import numpy as np
from sklearn.datasets import make_blobs
from uns_package.KMEDOIDS import KMEDOIDS

X,y=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1,center_box=(-10.0,10.0),shuffle=True,random_state=1,)
k_medoids = KMEDOIDS(n_clusters=4, check_convergence=True, distance='euclidean')
k_medoids.fit(X)
labels = k_medoids.predict(X)
print(labels)