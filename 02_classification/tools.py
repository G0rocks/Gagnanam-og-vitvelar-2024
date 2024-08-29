from typing import Union

import numpy as np
import sklearn.datasets as datasets


def load_iris():
    '''
    Load the iris dataset that contains N input features
    of dimension F and N target classes.

    Returns:
    * inputs (np.ndarray): A [N x F] array of input features
    * targets (np.ndarray): A [N,] array of target classes
    '''
    iris = datasets.load_iris()
    return iris.data, iris.target, [0, 1, 2]


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    
    features: The data points, an n X k array where n is the number of data points and k is how many dimensions each point has
    targets:    An n X 1 array which tells which class each feature belongs to
    train_ratio:    Value from 0 to 1 (both included). How high of a percentage ratio is used for the training data. 0.8 by default
    '''
    # Shuffle features and targets with the same random shuffle
    # Create p, a permutated (shuffled) array with n values
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    # Find where the split between training data and test data is.
    split_index = int(features.shape[0] * train_ratio)

    # Split data into training and testing set. 2 different ways, 1 for 1 dimensional data and 1 for 2 dimensional data
    if len(features.shape) > 1:
        train_features  =   features[0:split_index, :]
        train_targets   =   targets[0:split_index]
        test_features   =   features[split_index:, :]
        test_targets    =   targets[split_index:]
    else:
        train_features  =   features[0:split_index]
        train_targets   =   targets[0:split_index]
        test_features   =   features[split_index:]
        test_targets    =   targets[split_index:]

    # Returned split training and testing sets
    return (train_features, train_targets), (test_features, test_targets)
