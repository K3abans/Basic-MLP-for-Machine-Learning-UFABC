# Basic MLP for Machine Learning UFABC

This project demonstrates how to use scikit-learn's `MLPClassifier` on a synthetic dataset generated with `make_classification`.

## How it works

- Generates a dataset with 1000 samples and 50 features (5 informative).
- Splits the data into training and test sets.
- Standardizes the features.
- Trains a Multilayer Perceptron (MLP) classifier.
- Evaluates and prints the test accuracy.

## Requirements

- Python 3.x
- scikit-learn
- numpy

Install dependencies with:

```bash
pip install scikit-learn numpy
```

## Usage

Run the script:

```bash
python MLP.py
```