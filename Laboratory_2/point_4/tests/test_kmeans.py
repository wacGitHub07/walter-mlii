from math import sqrt, floor
import numpy as np
from sklearn.datasets import make_blobs
from uns_package.KMEANS import KMEANS

X,y=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1,center_box=(-10.0,10.0),shuffle=True,random_state=1,)
kmeans = KMEANS(n_clusters=4, init='random', random_state=1234)
kmeans.fit(X)
labels = kmeans.predict(X)
print(labels)
print(kmeans.centroids)