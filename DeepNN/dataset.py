import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def load_data():
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test