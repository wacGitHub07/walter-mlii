# LDA (Linear Discriminant Analisis)


Linear Discriminant Analysis (LDA) is a supervised classification technique that seeks to find a linear combination of features that maximizes the separation between classes in a training data set. As we have seen with PCA, LDA can be used to reduce the dimensionality of a dataset while keeping the most relevant information.

- PCA seeks to find a low-dimensional representation of the data that maximizes the total variance of the input data, while LDA seeks to find a low-dimensional representation of the data that maximizes the separation between classes in a labeled data set.

- PCA is an unsupervised learning technique, while LDA is a supervised learning technique that uses class labels to guide dimensionality reduction.

- PCA uses all the features in the input dataset, while LDA uses only a subset of features that better separate the classes. In other words, LDA is a feature selection technique as well as a dimensionality reduction technique.


### How works?
Let's consider an example where we have two classes in a 2-D plane having an X-Y axis, and we need to classify them efficiently. As we have already seen in the above example that LDA enables us to draw a straight line that can completely separate the two classes of the data points. Here, LDA uses an X-Y axis to create a new axis by separating them using a straight line and projecting data onto a new axis.

Hence, we can maximize the separation between these classes and reduce the 2-D plane into 1-D.

To create a new axis, Linear Discriminant Analysis uses the following criteria:
- It maximizes the distance between means of two classes.
- It minimizes the variance within the individual class.

Using the above two conditions, LDA generates a new axis in such a way that it can maximize the distance between the means of the two classes and minimizes the variation within each class.

In other words, we can say that the new axis will increase the separation between the data points of the two classes and plot them onto the new axis.


### How to Prepare Data for LDA?:

- Classification Problems. This might go without saying, but LDA is intended for classification problems where the output variable is categorical. LDA supports both binary and multi-class classification.

- Gaussian Distribution. The standard implementation of the model assumes a Gaussian distribution of the input variables. Consider reviewing the univariate distributions of each attribute and using transforms to make them more Gaussian-looking (e.g. log and root for exponential distributions and Box-Cox for skewed distributions).

- Remove Outliers. Consider removing outliers from your data. These can skew the basic statistics used to separate classes in LDA such the mean and the standard deviation.

- Same Variance. LDA assumes that each input variable has the same variance. It is almost always a good idea to standardize your data before using LDA so that it has a mean of 0 and a standard deviation of 1.


### Real-world Applications of LDA
Some of the common real-world applications of Linear discriminant Analysis are given below:

- Face recognition is the popular application of computer vision, where each face is represented as the combination of a number of pixel values. In this case, LDA is used to minimize the number of features to a manageable number before going through the classification process. It generates a new template in which each dimension consists of a linear combination of pixel values. If a linear combination is generated using Fisher's linear discriminant, then it is called Fisher's face.

- In the medical field, LDA has a great application in classifying the patient disease on the basis of various parameters of patient health and the medical treatment which is going on. On such parameters, it classifies disease as mild, moderate, or severe. This classification helps the doctors in either increasing or decreasing the pace of the treatment.

- Customer Identification: In customer identification, LDA is currently being applied. It means with the help of LDA; we can easily identify and select the features that can specify the group of customers who are likely to purchase a specific product in a shopping mall. This can be helpful when we want to identify a group of customers who mostly purchase a product in a shopping mall.

- For Predictions: LDA can also be used for making predictions and so in decision making. For example, "will you buy this product” will give a predicted result of either one or two possible classes as a buying or not.

- In Learning : Nowadays, robots are being trained for learning and talking to simulate human work, and it can also be considered a classification problem. In this case, LDA builds similar groups on the basis of different parameters, including pitches, frequencies, sound, tunes, etc.

References:
- https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning
- https://www.javatpoint.com/linear-discriminant-analysis-in-machine-learning
- https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/
- https://www.cienciadedatos.net/documentos/28_linear_discriminant_analysis_lda_y_quadratic_discriminant_analysis_qda
