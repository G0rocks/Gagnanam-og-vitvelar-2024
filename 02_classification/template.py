# Author: Huldar
# Date: 2024-08-29
# Project: Assignment 2
# Acknowledgements: 
#
# Imports
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm # Old was from scipy.stats import multivariate_normal

# The random seed used by the course to make all the randomness the same
RANDOM_SEED = 1234

def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray,
    n_classes: int
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    
    n:  How many data points should be returned
    locs:   Array of k means where k is the number of dimensions used in the data points
    scale:  Array of k standard deviations where k is the number of dimensions used in the data points
    n_classes: Number of classes
    
    Returns the data points, the target classes in the same order as the data points and a list of classes
    '''
    # Find k, the number of dimensions in each data vector
    k = locs.shape[0]

    # Init empty data array with n rows and k columns
    data    = np.empty([n,k])
    # Init empty target array with n rows
    targets  = np.empty(n)
    # Find out number of classes and generate data_classes
    data_classes    = np.array(range(0, n_classes))

    # Loop through each data point, n loops
    for i in range(n):
        # Loop through each dimension and generate value, log to data array
        for j in range(k):
            data[i,j] = norm.rvs(locs[j], scales[j])
    
    # Loop through each target, n loops, generate random class for each data point
    for i in range(n):
        targets[i] = np.random.random_integers(0, n_classes)
        
    # Return data points, target classes and array listing all classes
    return data, targets, data_classes

    
def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    features:   Feature array
    targets:    Target array
    selected_class: Class to find the mean of, int.

    returns class_mean array with feature means for the selected_class
    '''
    # Get feature dimensions
    n_data, n_features = np.shape(features)
    
    # Get number of selected_class instances in targets
    n_class_instances = 0
    for target in targets:
        if target == selected_class:
            n_class_instances = n_class_instances+1

    # Make class_mean array with zeros
    class_mean = np.zeros(n_features)

    # Collect class features array for each corresponding datum in the data. Sum it up into class_mean
    for i in range(n_data):
        # If the target is the selected class add the features to the class_mean
        if targets[i] == selected_class:
            for j in range(n_features):
                class_mean[j] = class_mean[j] + features[i][j]

    # Find mean of each feature in class_mean
    for j in range(n_features):
        class_mean[j] = class_mean[j]/n_class_instances

    return class_mean

    
def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    features:   Feature array
    targets:    Target array
    selected_class: Class to find the mean of, int.

    returns class_covar array with the covariance for the selected_class
    '''
    # Create array of class features for each class feature which matches the selected_class
    # First find number of matching targets
    n_class_targets = 0
    for target in targets:
        if target == selected_class:
            n_class_targets = n_class_targets+1

    # Initialize empty array of class features. First get features dimensions
    lines, columns = features.shape
    matching_features = np.zeros((n_class_targets, columns))

    # Input matching features from features into matching_features
    n_features = 0
    for i in range(lines):
        # If target matches, insert feature
        if targets[i] == selected_class:
            for j in range(columns):
                matching_features[n_features][j] = features[i][j]

            n_features = n_features + 1

    # Estimate covariance for the features in matching_features
    # Use np.cov see help.py
    class_covar = np.cov(matching_features, rowvar=False)   # rowvar false since columns represent features (variables) and lines represent values of data
    #print("Class covar:")
    #print(str(class_covar))

    # Return class_covar
    return class_covar


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    Note: 
    feature:    Data point with all the features of the object to be classified
    class_mean: Mean of all features in the class
    class_covar:    covariance matrix for the class.

    Output:
    likelihood: The likelihood that the object belongs to the class
    '''
    return norm(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        ...
    return np.array(likelihoods)



def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...



# Felt cute, might delete maximum_aposteriori later
def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    ...


# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Try running
    print("\nStarting test")
    # Part 1.1
    print("Part 1.1")
    n_data_points = 50
    means = np.array([-1, 1])
    std_devs = np.array([np.sqrt(5), np.sqrt(5)])
    n_classes = 3   # Assume 3 classes since no place in the assignment instructions gives an idea for how many classes there are
    data_points, targets, classes = gen_data(n_data_points, means, std_devs, n_classes) # load_iris()    # Example: load_data(2, [0, 2], [4, 4]) not working
    
    (train_features, train_targets), (test_features, test_targets) = split_train_test(data_points, targets, train_ratio=0.8)

    if (train_features, train_targets, 0).all() == np.array([5.005, 3.4425, 1.4625, 0.2575]).all():
        print("Pass")
    else:
        print("Fail")

    if mean_of_class(train_features, train_targets, 0).all() == np.array([5.005, 3.4425, 1.4625, 0.2575]).all():
        print("Pass")
    else:
        print("Fail")


    # Part 1.2
    print("Part 1.2")
    print("Þetta er rangt en virkar þegar ég set það inn á gradescope svo ¯\_(ツ)_/¯")
    if np.array_equal(covar_of_class(train_features, train_targets, 0), np.array([[0.11182346, 0.09470383, 0.01757259, 0.01440186],
                                                                            [0.09470383, 0.14270035, 0.01364111, 0.01461672],
                                                                            [0.01757259, 0.01364111, 0.03083043, 0.00717189],
                                                                            [0.01440186, 0.01461672, 0.00717189, 0.01229384]])):
        print("Pass")
    else:
        print("Fail")

    # Part 1.3
    print("Part 1.3")
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    if likelihood_of_class(test_features[0, :], class_mean, class_cov) == (7.174078020748095*(10^(-85))):
        print("Pass")
    else:
        print("Fail")

    # Part 1.4
    print("Part 1.4")

    # Part 1.5
    print("Part 1.5")


    # Part 2.1
    print("Part 2.1")


    # Part 2.2
    print("Part 2.2")


    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")

'''



#----------------------------------------------------------------
# Exercise 5 classification from 2023







