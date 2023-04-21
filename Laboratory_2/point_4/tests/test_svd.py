#%%
from uns_package.SVD import SVD
import numpy as np
import random

# Testing the svd class

#%%
m = np.random.randint(low=2, high=5)
n = np.random.randint(low=2, high=5)

randomlist = random.sample(range(0, m*n), m*n)
A = np.array(randomlist).reshape((m,n)).astype('float')
print(A)

#%%
svd = SVD(method='eig')
# %%
svd.fit(A)

# %%
u, sigma, vt = svd.transform()
# %%
print("U")
print(u)
print("Sigma")
print(sigma)
print("V")
print(vt)


Sigma = np.zeros((m, n))
dim = m
if m > n: dim = n
Sigma[:dim,:dim] = np.diag(sigma)
B = u.dot(Sigma.dot(vt))
np.round(B)
# %%
