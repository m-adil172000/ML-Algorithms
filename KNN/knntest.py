# DATA:  We are going to use the iris-dataset

#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Loading the dataset
iris_data = datasets.load_iris()
X, y = iris_data.data, iris_data.target


# Train-test-split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Let's look at the shape of our training data
print(f"Shape of training data: {X_train.shape}")

# Let's look at the first sample
print(f"First sample: {X_train[0]}")

# Let's look at the labels
print(f"Labels: {y_train}")
'''
From the labels, we can see that we have three labels - [0,1,2] which means
it is a three class classification problem.
'''

#Let's plot our data to get a better understanding
plt.figure(figsize=(5,5))
scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.colorbar(scatter, label='Species')
plt.show()

# Let's use our KNN algorithm to classify the data
from knn import KNN
knn = KNN(k=3)
knn.fit(X_train,y_train)

predictions = knn.predict(X_test)
print(f"Prediction: {predictions}, Labels: {y_test}")

# Let's see how accurate our model is
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


