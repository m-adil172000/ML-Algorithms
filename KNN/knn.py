import numpy as np


class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):

        #calculate euclidean distance pf x with all the other data points in X_train
        distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]

        #Sorting the distances in ascending order in order to get the nearest k neighbours
        k_indices = np.argsort(distances)[:self.k]

        #form the above indices, we need to find the labels of those k-nearest neighbours
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #Now we return the most common labels
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common
