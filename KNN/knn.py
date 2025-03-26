# K-Nearest Neighbors (KNN) Classifier from scratch using NumPy
# This model stores training data and predicts labels based on the majority class 
# of the k-nearest neighbors using Euclidean distance.
import numpy as np 
from collections import Counter

class KNN:
    def __init__(self, k = 3):
        self.k = k 
        
    def fit(self, X, Y):
        self.X_train = X 
        self.Y_train = Y
        
    def eucledian_distance(self, X, X1):
        return np.sqrt(np.sum((X, X1) ** 2))  
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [self.eucledian_distance(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN model
knn = KNN(k=3)
knn.fit(X_train, y_train)