from dataset import load_data
from deepnn import DeepNNScratch
import numpy as np

X_train, X_test, y_train, y_test = load_data()

nn = DeepNNScratch(input_size=2, output_size=1, hidden_size=4, iteration=2000, learning_rate=0.01)

print("Training the model...")
nn.fit(X_train, y_train)

y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

train_accuracy = np.mean(y_pred_train == y_train) * 100
test_accuracy = np.mean(y_pred_test == y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")