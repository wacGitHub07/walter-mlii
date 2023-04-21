#%%
from uns_package.TSNE import TSNE
import numpy as np
import random

# Testing the tsne  class

#%%
m = np.random.randint(low=5, high=10)
n = np.random.randint(low=5, high=10)
m,n

randomlist = random.sample(range(0, m*n), m*n)
X = np.array(randomlist).reshape((m,n)).astype('float')
print(X)

#%%
tsne = TSNE(n_components=2)

#%%
tsne_data = tsne.fit_transform(X)
print(tsne_data)
# %%
