# Create Test Datasets Using Scikit-learn
This guide explores Scikit-learn's functionalities for generating test datasets for various machine learning tasks.

Scikit-learn Datasets:

Scikit-learn provides a convenient datasets module with functions to generate sample datasets for different learning problems. This allows you to experiment with algorithms and visualize data distributions without requiring real-world datasets.

Classification Datasets
1. Blob-like Data with make_blobs:

This function generates datasets consisting of clusters or "blobs" in two-dimensional space. It's suitable for clustering problems:

Python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create blobs with 3 centers, 100 samples, and some cluster standard deviation
x, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=2)

# Plot the data points
plt.scatter(x[:, 0], x[:, 1], s=40, color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()


2. Moon-shaped Data with make_moons:

This function generates two crescent-shaped clusters, useful for binary classification problems with non-linear boundaries:

Python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Create moon-shaped data with 1000 samples and some noise
x, y = make_moons(n_samples=1000, noise=0.1)

# Plot the data points
plt.scatter(x[:, 0], x[:, 1], s=40, color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()


3. Concentric Circles with make_circles:

This function generates two concentric circles, representing a binary classification problem with a circular decision boundary:

Python
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Create concentric circles with 100 samples and some noise
x, y = make_circles(n_samples=100, noise=0.1)

# Plot the data points
plt.scatter(x[:, 0], x[:, 1], s=40, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.clf()


4. Customizing Classification Datasets:

These functions allow you to adjust parameters like the number of samples, number of clusters/classes, noise level, and random seed for reproducibility.

Regression Datasets
1. Linear Regression with make_regression:

This function generates a simple linear relationship between a single feature and a target variable.

Python
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Create a 1D regression dataset with 50 samples, noise, and random seed
x, y = make_regression(n_samples=50, n_features=1, noise=20, random_state=23)

# Plot the data points
plt.scatter(x, y)
plt.show()


2. Multi-feature Regression with make_sparse_uncorrelated:

This function generates a dataset with sparse, uncorrelated features for regression with multiple features:

Python
from sklearn.datasets import make_sparse_uncorrelated
import matplotlib.pyplot as plt

# Create a dataset with 100 samples, 4 features, and random seed
x, y = make_sparse_uncorrelated(n_samples=100, n_features=4, random_state=23)

# Plot each feature vs. target variable
plt.figure(figsize=(12, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.scatter(x[:, i], y)
    plt.xlabel('x' + str(i + 1))
    plt.ylabel('y')
plt.show()   



Scikit-learn offers various other functions for generating test datasets, including:

make_classification: General classification data with customizable parameters.
make_multilabel_classification: Multi-label classification data where each sample can belong to multiple classes.
make_friedman2: Non-linear regression dataset with multiple features.
By exploring these functionalities, you can effectively create test datasets for various machine learning tasks within your Python environment.
