from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def get_dataset():
    print('*** Downloading dataset...')
    return fetch_openml('mnist_784', version=1)


def separate_features_labels(dataset):
    return dataset['data'], dataset['target']


def convert_label_type(labels):
    return labels.astype(np.uint8)


def split_train_test(features, labels):
    return features.iloc[:60000], features.iloc[60000:], labels[:60000], labels[60000:]


def train_model(features, labels):
    print('*** Training model...')
    model = KNeighborsClassifier()
    model.fit(features, labels)
    return model


def evaluate_on_test_set(model, features, labels):
    print('*** Evaluating on test set...')
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f'Model accuracy: {accuracy}')
