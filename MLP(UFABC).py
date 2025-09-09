"""
MLP Classifier Example with Synthetic Data

This script demonstrates how to use scikit-learn's MLP (UFABC)
on a synthetic classification dataset.
"""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=42
    )

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define and train the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000)
    mlp.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("****************************")
    print(f"-> Test accuracy: {accuracy:.2f}, hidden layers: {mlp.hidden_layer_sizes}, max_iter: {mlp.max_iter}")
    time.sleep(1)
    print("****************************")
    time.sleep(2)
    print(f"-> Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
    print("****************************")
    time.sleep(1)

if __name__ == "__main__":
    main()