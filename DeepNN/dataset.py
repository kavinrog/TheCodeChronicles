"""
dataset.py - Data Loader Function for Binary Classification (Moons Dataset)

This function generates a synthetic 2D classification dataset using `make_moons`,
adds noise for complexity, and performs the following steps:
- Scales features using StandardScaler (zero mean and unit variance)
- Splits data into training and testing sets (80/20 split)
- Reshapes the target labels for compatibility with neural network training

Returns:
    X_train, X_test, y_train, y_test: Preprocessed data ready for training and evaluation
"""
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