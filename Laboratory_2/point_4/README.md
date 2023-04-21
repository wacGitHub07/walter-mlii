# Unsupervised Package
## Walter Arboleda Castañeda - Machine Learnig II - UdeA - 2023

**Description:**

This package contains the implementation of three algorithms for dimensionality reduction and clustering :

Dimensionality reduction
- SVD
- PCA
- TSNE

Clustering
- KMEANS
- KMEDOIDS

**Details:**
- SVD algorithm was implemented with two methos: power iterate as principal method ans calculating the right and left eigenvalues and vectors as auxiliar method. This is because in tests the power iterate method only worked with sqare matrices.
- PCA was computed by centralizing the data,  calculating  eigenvalues and vectors and covariance matrix 
- STNE was computed following the traditional definition
- SVD and PCA have fit(), transform() and fit_transform() methods, TSNE only has fit_transform() method due its algorithm characteristics 
- KMEANS was computed with three methods to initialize vectors (random, kmeans++, naive_sharing) and a error tolerance to control the convergence
- KMEDOIDS was computed with two methods to calcule the distance (euclidean, manhattan).

**Dependences:**

The package is builing with poetry with the next dependences:
- python = "^3.10"
- scikit-learn = "^1.2.2"

**Usage:**

In the dist folder you can find the .whl and .gz files for package instalation
- uns_package-0.1.0-py3-none-any.whl (windows)
- uns_package-0.1.0.tar.gz (linux, mac)

```python
#Example:

# X ndarray with input data

# SVD
from uns_package.SVD import SVD
svd = SVD(method='eig')
svd.fit(A)
u, sigma, vt = svd.transform()
print("U")
print(u)
print("Sigma")
print(sigma)
print("V")
print(vt)

#PCA
from uns_package.PCA import PCA
pca = PCA(n_components=2)
pca.fit(X)
pca = PCA(n_components=2)
pca.fit(X)
print(pca.cumulative_variance)
print(pca.explained_variance)
pca_data = pca.transform(X)
print(pca_data)

#TSNE
from uns_package.TSNE import TSNE

tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(X)
print(tsne_data)

# KMEANS
from sklearn.datasets import make_blobs
from uns_package.KMEANS import KMEANS

X,y=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1,center_box=(-10.0,10.0),shuffle=True,random_state=1,)
kmeans = KMEANS(n_clusters=4, init='random', random_state=1234)
kmeans.fit(X)
labels = kmeans.predict(X)
print(labels)
print(kmeans.centroids)

# KMEDOIDS
from sklearn.datasets import make_blobs
from uns_package.KMEDOIDS import KMEDOIDS

X,y=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1,center_box=(-10.0,10.0),shuffle=True,random_state=1,)
k_medoids = KMEDOIDS(n_clusters=4, check_convergence=True, distance='euclidean')
k_medoids.fit(X)
labels = k_medoids.predict(X)
print(labels)


```
