from sklearn.datasets import fetch_openml
import numpy as np


def get_dataset():
    return fetch_openml('mnist_784', version=1)


def separate_features_labels(dataset):
    return dataset['data'], dataset['target']


def convert_label_type(labels):
    return labels.astype(np.uint8)


def split_train_test(features, labels):
    return features.iloc[:60000], features.iloc[60000:], labels[:60000], labels[60000:]

