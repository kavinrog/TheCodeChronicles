import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class SVM:
    
    def __init__(self, learning_rate=0.001, iteration=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None 
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        
        # Ensure labels are in {-1, 1}
        y_transformed = np.where(y <= 0, -1, 1)

        for _ in range(self.iteration):
            for idx, xi in enumerate(X):  # xi is a single sample
                condition = y_transformed[idx] * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_transformed[idx] * xi)
                    self.bias -= self.learning_rate * y_transformed[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)
    
np.random.seed(42)
num_samples = 100

X_class1 = np.random.randn(num_samples // 2, 2) - 2
y_class1 = np.full((num_samples // 2,), -1)

X_class2 = np.random.randn(num_samples // 2, 2) + 2
y_class2 = np.full((num_samples // 2,), 1)

X = np.vstack((X_class1, X_class2))
y = np.hstack((y_class1, y_class2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVM()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='red', label='Class -1')
plt.scatter(X_class2[:, 0], X_class2[:, 1], color='blue', label='Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Toy Dataset for SVM")
plt.legend()
plt.show()