#%%
# Testing the pca class

from uns_package.PCA import PCA
import numpy as np
import random

#%%
m = np.random.randint(low=5, high=10)
n = np.random.randint(low=5, high=10)
m,n

randomlist = random.sample(range(0, m*n), m*n)
X = np.array(randomlist).reshape((m,n)).astype('float')
print(X)

#%%
pca = PCA(n_components=2)
pca.fit(X)
#%%
print(pca.cumulative_variance)
print(pca.explained_variance)
# %%
pca_data = pca.transform(X)
print(pca_data)
# %%
