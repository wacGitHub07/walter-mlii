# walter-mlii-lab1
This repository contains the solution to lab I of the machine learning II course, UdeA 2023.

**Auhor:** Walter Arboleda Castañeda

The solution to the points is organized in folders as follows:

- **point 1**: Contains the solution to: rank, trace, determinant, eigenvalues and eigenvectors.
- **point 2**: Contains the folder with the profile pictures of the cohort and the solution to resize image, change to gray scale and calculate the difference between my personal photo vs the average of the cohort.
- **point 3**: contains the unsupervised python package builded with poetry. Inside it there are the implementations to SVD, PCA and TSNE algorthmns and the .whl file to instalations.
- **point 4**: Contains the applications of SVD over my personal picture, and the iterative reconstructions varying the singular values.
- **point 5_6_7**: These 3 points was implemented in the same notebook due i need to compare the performance betwwen models applying my personal PCA algorithm and scikit-learn PCA trainig a logistic regression.
- **point 8**: Contains a basic description of methods to make traditional PCA more robust, and a little example of how implement them.
- **point 9**: Contains a description of UMAP and a example of how implement it.
- **point 10**: Contains a description of LDA and a example of how implement it.
- **point 11**: Contains the code to deploy an api created with flask and flask-restfull, this api exposes and endpoint to predict the iris dataset classes. The model (KNN) was trained with 2 components generated with my personal PCA algorithm.

The requirements.txt files contains all libraries used.