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

# Reset random seed
np.random.seed(RANDOM_SEED)

def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
    # n_classes: int
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    
    n:  How many data points should be returned
    locs:   Array of means. c X k array where c is the number of classes and k is the number of dimensions used in the data points. Here k=1 for simplification
    scale:  Array of standard deviations. c X k array where c is the number of classes and k is the number of dimensions used in the data points. Here k=1 for simplification
    # n_classes: Number of classes
    
    Returns the data points, the target classes in the same order as the data points and a list of classes
    '''
    # Find k, the number of dimensions in each data vector
    k = 1   # k = 1 because this is a simplified case
    
    # Find out the number of classes
    n_classes = locs.shape[0]

    # Init empty data array with n rows and k columns
    data    = np.empty([n,k])
    # Init empty target array with n rows
    targets  = np.empty(n, dtype=int)
    # Generate data_classes
    data_classes    = np.array(range(0, n_classes))

    # Loop through each data point, n loops
    for i in range(n):
        # Check which class it belongs to and assign data point class to targets
        targets[i] = np.random.randint(n_classes)
        # Loop through each dimension and generate value, log to data array
        for j in range(k):
            data[i,j] = norm.rvs(locs[targets[i]], scales[targets[i]])
            
    # Return data points, target classes and array listing all classes
    return data, targets, data_classes


def _get_num_class_in_targets(targets: np.ndarray, sel_class: int) -> int:
    '''
    Gets the number of instances a given class appears in the targets
    targets: The target array
    sel_class: The selected class
    '''
    n_class_instances = 0
    for target in targets:
        if target == sel_class:
            n_class_instances = n_class_instances+1
    return n_class_instances

def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    
    features:   Feature array, contains all the data points
    targets:    Target array
    selected_class: Class to find the mean of, int.

    returns class_mean array with feature means for the selected_class
    '''
    # Get feature dimensions
    # Try catch because of gradescope problems
    try:
        n_data, n_features = np.shape(features)
        gradescope = False
    except:
        n_data = np.shape(features)[0]
        n_features = 1
        gradescope = True
    
    # Get number of selected_class instances in targets
    n_class_instances = _get_num_class_in_targets(targets, selected_class)

    # Make class_mean array with zeros
    class_mean = np.zeros(n_features)

    # Collect class features array for each corresponding datum in the data. Sum it up into class_mean
    for i in range(n_data):
        # If the target is the selected class add the features to the class_mean
        if targets[i] == selected_class:
            for j in range(n_features):
                if gradescope:
                    class_mean[j] = class_mean[j] + features[i]
                else:
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
    # Get number of selected_class instances in targets
    n_class_instances = _get_num_class_in_targets(targets, selected_class)

    # Initialize empty array of class features. First get features dimensions    
    # Try catch because of gradescope problems
    try:
        lines, columns = np.shape(features)
        gradescope = False
    except:
        lines = np.shape(features)[0]
        columns = 1
        gradescope = True
    
    matching_features = np.zeros((n_class_instances, columns))

    # Input matching features from features into matching_features
    n_features = 0
    for i in range(lines):
        # If target matches, insert feature
        if targets[i] == selected_class:
            for j in range(columns):
                if gradescope:
                    matching_features[n_features][j] = features[i]
                else:
                    matching_features[n_features][j] = features[i][j]
            # Update number of features
            n_features = n_features + 1

    # Estimate covariance for the features in matching_features
    # Use np.cov see help.py
    class_covar = np.cov(matching_features, rowvar=False).reshape(-1)   # rowvar false since columns represent features (variables) and lines represent values of data
    # print("Class covar:")
    # print(str(class_covar))

    # Return class_covar
    return class_covar

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn from a multivariate normal distribution, given the mean and covariance of the distribution.

    feature:    Data point with all the features of the object to be classified
    class_mean: Mean of all features in the class
    class_covar:    covariance matrix for the class.

    Output:
    likelihood: The likelihood that the object belongs to the class
    '''
    # Find class standard deviation from covariance
    # Init class_std_dev
    class_std_dev = np.zeros(class_covar.shape[0])
    # For each variance value on the diagonal of the covariance matrix, take the square root
    for i in range(class_covar.shape[0]):
        class_std_dev[i] = np.sqrt(class_covar[i])
    
    # Find class likelihood
    class_likelihood = norm(class_mean, class_std_dev).pdf(feature)
    return class_likelihood

def maximum_likelihood(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    test_data: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_data by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_data.shape[0] x len(classes)] shaped numpy
    array
    '''
    # Init means and covariances (covs) for all classes
    means, covs = [], []
    # For each class, calculate the mean and covariance in the training set, append to means and covs
    for class_label in classes:
        # Find and append mean of class to means
        means.append(mean_of_class(train_data, train_targets, class_label))
        # Find and append covariance of class to covs
        covs.append(covar_of_class(train_data, train_targets, class_label))
        
    # Init likelihoods a [c X n] array where c are the number of classes and n are the number of data points in the test set
    # Find c
    c = len(classes)
    # Find n
    n = test_data.shape[0]
    
    # Init likelihoods
    likelihoods = np.empty((n, c))
    # For each data point in the test set and each class, calculate likelihood of data point belonging to that specific class. Log value in likelihoods
    # Loop through test data points
    for i in range(n):
        # Loop through each class
        for j in range(c):
            # Calculate likelihood of data point belonging to class, log in likelihoods
            likelihoods[i, j] = likelihood_of_class(test_data[i], means[j], covs[j])
        
    # Return likelihoods
    return likelihoods

def predict(likelihoods: np.ndarray) -> np.ndarray:
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for which class each datapoint belongs to by choosing the
    highest likelihood.

    likelihoods:    The maximum likelihood for each test point as found in maximum_likelihood

    Returns: A [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    # Find num_datapoints
    num_datapoints = likelihoods.shape[0]
    # Find num_classes
    num_classes = likelihoods.shape[1]
    
    # Init predictions, an array with num_datapoints values
    predictions = np.zeros(num_datapoints)
    
    # For each datapoint predict which class it will belong to based on the likelihoods
    for i in range(num_datapoints):
        # Init/reset max_likelihood to -1
        max_likelihood = -1
        
        # Init/reset class_memory - Used to keep track of which class has the highest likelihood
        class_memory = -1
        
        # For each class, compare likelihood to the max_likelihood of this datapoint
        for j in range(num_classes):
            # If the likelihood of this data point belonging to this class is higher than max_likelihood, update max_likelihood with the likelihood value and log the current class being checked to class_memory
            if likelihoods[i, j] > max_likelihood:
                max_likelihood = likelihoods[i, j]
                class_memory = j
                    
        # Log class_memory to predictions for this datapoint
        predictions[i] = class_memory

    # Return predictions
    return predictions

def predict_accuracy(predictions: np.ndarray, targets: np.ndarray)   -> float:
    '''
    Compares how many of the predictions were correct, n_correct, returns n_correct / n_targets
    Note: the shape of the inputs must be the same.
    
    predictions:    The class predictions
    targets:        The actual correct classes for each datapoint
    
    returns accuracy unless inputs are not the same shape, then return None
    '''
    # Validate inputs to be of same shape, if not, return None
    if predictions.shape[0] != targets.shape[0]:
        return None
    
    # init n_correct
    n_correct = 0
    
    # Find number of data_points, n_targets
    n_targets = targets.shape[0]
    
    # Loop through each data point and compare them
    for i in range(n_targets):
        # Compare prediction against target, if the prediction and the target are the same class, increase n_correct by 1
        if predictions[i] == targets[i]:
            n_correct = n_correct + 1

    # Return accuracy
    return n_correct/n_targets

# Felt cute, might delete maximum_aposteriori later
def maximum_aposteriori(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    test_data: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_data by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_data.shape[0] x len(classes)] shaped numpy
    array
    '''
    ...

# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    # Try running
    print("\nStarting test")
    # Part 1
    print("Part 1 - Generate data")
    n_data_points = 50
    means = np.array([-1, 1])
    std_devs = np.array([np.sqrt(5), np.sqrt(5)])
    n_classes = means.shape[0]
    data_points, targets, classes = gen_data(n_data_points, means, std_devs) #, n_classes) # load_iris()    # Example: load_data(2, [0, 2], [4, 4]) not working
    # Split data
    train_ratio = 0.8
    (train_data, train_targets), (test_data, test_targets) = split_train_test(data_points, targets, train_ratio)

    '''
    if (train_data, train_targets, 0).all() == np.array([5.005, 3.4425, 1.4625, 0.2575]).all():
        print("Pass")
    else:
        print("Fail")

    if mean_of_class(train_data, train_targets, 0).all() == np.array([5.005, 3.4425, 1.4625, 0.2575]).all():
        print("Pass")
    else:
        print("Fail")
    '''

    # Part 2
    print("Part 2 - Plot")
    # Generate random colors which is a n x 3 list where [n][0] signifies RED, [n][1] signifies GREEN and [n][2] signifies BLUE in the RGB code
    colors = np.empty([n_classes, 3])
    for i in range(n_classes):
        # Generate color
        colors[i][0] = np.random.random()    # RED
        colors[i][1] = np.random.random()    # GREEN
        colors[i][2] = np.random.random()    # BLUE

    # Loop through all data points, add each one to plot, depending on target, set marker
    for i in range(n_data_points):
        x = data_points[i]
        y = 0   # Set to zero since data_points is a n X 1 array in this simplified version
        
        # Get target class and corresponding color
        color_index = targets[i]
        color = (colors[color_index][0], colors[color_index][1], colors[color_index][2])
        plt.scatter(x, y, color = color, label = "Class " + str(targets[i]))

    # Make a legend to indicate which color is which class, credit to Homayoun Hamedmoghadam from https://stackoverflow.com/questions/70240576/avoid-duplicate-labels-in-matplotlib-with-x-y-of-form
    # Get legend handles and their corresponding labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # zip labels as keys and handles as values into a dictionary, so only unique labels would be stored 
    dict_of_labels = dict(zip(labels, handles))

    # use unique labels (dict_of_labels.keys()) to generate your legend
    plt.legend(dict_of_labels.values(), dict_of_labels.keys())
    
    # Label axis
    plt.xlabel("x1") # Add ", fontsize = #" to control fontsize
    plt.ylabel("x2")
    
    # Show or save plot
    # plt.show()
    plt.savefig(".\\Gagnanam-og-vitvelar-2024 git repo\\02_classification\\2_1.png")
    
    
    # Part 3
    print("Part 3 - Class mean")
    # Find class mean in the training data
    given_class = 0
    class_mean = mean_of_class(train_data, train_targets, given_class)

    # Part 4
    print("Part 4 - Class covariance")
    given_class = 0
    class_cov = covar_of_class(train_data, train_targets, given_class)

    # Part 5
    print("Part 5 - Class likelihood")
    class_likelihood = likelihood_of_class(test_data[0, :], class_mean, class_cov)

    # Part 6
    print("Part 6 - Maximum likelihood")
    max_likelihood = maximum_likelihood(train_data, train_targets, test_data, classes)

    # Part 7
    print("Part 7 - Predict likelihoods")
    class_prediction = predict(max_likelihood.transpose())
    print(class_prediction)

    # Part 8
    print("Part 8 - Comparison")
    # Create a new dataset with 50 datapoints and N (−4 , sqrt(2)) and N(4 ,sqrt(2)).
    n_data_points_8 = 50
    means_8 = np.array([-4, 4])
    std_devs_8 = np.array([np.sqrt(2), np.sqrt(2)])
    data_points_8, targets_8, classes_8 = gen_data(n_data_points_8, means_8, std_devs_8)
    '''
    (Question A) Compare the accuracy of both datasets, if the results are different, what explains the difference?
    Play around with the number of datapoints, the mean and standard deviation of the normal distributions.
    '''
    # To compare the accuracy, we must first measure the accuracy of the first dataset, then calculate the accuracy for the new dataset and then compare
    acc1 = predict_accuracy(class_prediction, test_targets)
    print("Accuracy for old dataset: " + str(acc1*100) + "%")
    
    # Find accuracy for data from section 8
    # Split data
    (train_data_8, train_targets_8), (test_data_8, test_targets_8) = split_train_test(data_points_8, targets_8, train_ratio)
    print("New dataset test_targets: " + str(test_targets_8))

    # Find mean
    given_class_8 = 0
    class_mean_8 = mean_of_class(train_data_8, train_targets_8, given_class_8)

    # Find covariance
    given_class_8 = 0
    class_cov_8 = covar_of_class(train_data_8, train_targets_8, given_class_8)

    # Find class likelihood
    class_likelihood_8 = likelihood_of_class(test_data_8[0, :], class_mean_8, class_cov_8)

    # Find max likelihood
    max_likelihood_8 = maximum_likelihood(train_data_8, train_targets_8, test_data_8, classes_8)

    # Predict classes
    class_prediction_8 = predict(max_likelihood_8.transpose())
    print("New dataset class predictions: " + str(class_prediction_8))

    # Find accuracy of predictions from section 8
    acc8 = predict_accuracy(class_prediction_8, test_targets_8)
    print("Accuracy for new dataset: " + str(acc8*100) + "%")
    diff = 100*(acc8-acc1)
    if diff == 0:
        diff_str = "Is exactly the same as from the old dataset"
    elif diff < 0:
        diff_str = "Is {:.2f}% less than from the old dataset".format(-diff)
    else:
        diff_str = "Is {:.2f}% more than from the old dataset".format(diff)

    print(diff_str)
    
    '''
    (Question B) What happens when you change the:
        number of datapoints
        mean of the normal distributions
        standard deviation of the normal distributions
    Explain in a few words how and why each of these affect the accuracy of your model
    '''

    print("Writing to 8_1.txt file")
    text_answer = \
        "For part A:\n\
            The results were different, the accuracy for the new data was 100%, 40% more than the accuracy for the old data.\n\
            For part B:\n\
            When I increased the number of data points for the old dataset to 500 instead of 50, the accuracy went up to 71% and was only 29% less\
            than the new dataset.\n\
            But when I reduced the number of points for the new set to 25, the accuracy was still 100%.\n\
            Changing the new mean to -2 and +2 resulted in an accuracy of 70% for the new dataset but in that case the old one got 80% accuracy.\n\
            I expected the standard deviation to have the most effect so I increased it to sqrt(10) for the new dataset, expecting lower accuracy.\n\
            but ended up with 90% accuracy that time and 80% for the old dataset.\n\
            To explain why these things happen I think I would need to do this very often or with the same randomness every time.\n\
            Realising I forgot to set the random seed in the beginning of my code I run it again and default to the old acc being 80% and the new being 70%.\n\
            With 500 old data points, new std_dev at sqrt(10), the old acc is 62% and the new 90%.\n\
            With the new mean being -2 and +2, new std_dev at sqrt(10), the old acc is 80% still and the new acc stayed at 70%.\n\
            With the new std_dev at sqrt(2) the old acc is 80% still and the new acc is 100%.\n\
            That's in line with my expectations that lower deviation results in higher accuracy, probably because the classes don't overlap as much, meaning the maximum likelihood is likelier to be correct.\n\
            The same is applicable to the mean, if the means are further from each other then it's unlikelier that the deviation will cause the values to overlap.\n\
            I'm starting to think https://en.meming.world/images/en/a/aa/Something_of_a_Scientist.jpg\n\n\
            Note: About the submission, now that I've updated the data_gen function to be in accordance with how I was told to have it, the gradescope gives me errors about not having enough values to unpack things.\n\
            This was not a problem in my previous version which had n_classes as an input for the gen_data function.\n\
            Note: For next year, add more explicitness about the number of classes and number of columns in the features array there should be."
    with open('.\\Gagnanam-og-vitvelar-2024 git repo\\02_classification\\8_1.txt', 'w') as f:
        f.write(str(text_answer))
    print("File updated")


    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")

'''