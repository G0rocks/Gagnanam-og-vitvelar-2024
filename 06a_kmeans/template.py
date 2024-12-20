# Author: Huldar
# Date: 2024-10-23
# Project: Assignment 6a
# Acknowledgements: 
#

# Imports
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results

def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in X and Mu.

    Inputs:
    X : A [N x f] array of samples
    Mu : A [K x f] array of prototypes

    outputs:
    dist : A [N x K] array of euclidian distances where dist[i, j] is the euclidian distance between X[i, :] and Mu[j, :]
    '''
    # Find N, K and f
    N, f = X.shape
    K = Mu.shape[0]
    
    # Initialize dist as zeros of size (n x k)
    dist = np.zeros([N,K])
    
    # For each each value in dist, calculate distance between X and Mu
    for i in range(N):  # For each data sample
        for j in range(K):  # For each prototype vector
            # Calculate distance
            dist[i,j] = np.linalg.norm(X[i,:]-Mu[j,:])            
        # End for j
    # End for i
    
    # Return dist
    return dist

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input:
    dist : A [N x K] array of distances

    Returns:
    R : A [N x K] array where R[i, j] is 1 if sample i is closest to prototype j and 0 otherwise.
    '''
    # Init R as zeros of size (N x K)
    N, K = dist.shape
    R = np.zeros([N,K],dtype=int)
    
    # Loop through each row of dist and set the value in r to 1 for the lowest corresponding distance in dist
    for n in range(N):
        # Reset or initialize index and min_dist
        index = -1
        min_dist = float('inf') # Set as high as possible
        # Loop through each column of dist to check the distances
        for k in range(K):
            # Check if distance is smaller than the min_dist found so far, if so, update min_dist and index
            if dist[n,k] < min_dist:
                min_dist = dist[n,k]
                index = k
        # end for k
        
        # Set value for R at index as 1
        R[n,index] = 1
    # end for n
    
    # Return R
    return R

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given arrays of indicators and distances.
    Based on equation 15.1 from Bishop

    Inputs:
    R : [N x K] array where R[i, j] is 1 if sample i is closest to prototype j and 0 otherwise.
    dist : [N x K] array of distances

    output:
    j_hat : The value of the objective function
    '''
    # Get linear combination of R and dist for each row in dist
    # Find minimum value
    # j = np.min(dist.dot(R.transpose()))
    N, K = dist.shape
    
    j_hat = 0
    
    for n in range(N):
        for k in range(K):
            j_hat = j_hat + R[n][k]*dist[n][k] #*dist[n][k]

    '''
    # How Danielle did it
    total_dist = np.sum(R*dist)
    average_dist = total_dist / R.shape[0]
    '''
    
    # return j_hat
    return j_hat/N

def update_Mu(Mu: np.ndarray, X: np.ndarray, R: np.ndarray) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current prototypes, samples and indicators.
    Equation from instructions:
    M-step, mu_k = sum over n {r_nk * x_n} / sum over n {r_nk}

    Inputs:
    Mu: A [K x f] array of current prototypes.
    X : A [N x f] array of samples.
    R : A [N x K] array of indicators.

    Returns:
    mu_new : A [K x f] array of updated prototypes.
    '''
    # Find N, K and f
    N = X.shape[0]
    K = Mu.shape[0]
    f = Mu.shape[1]
    
    # Initialize updated mu array, mu_new
    mu_new = np.zeros([K, f])
    
    # For each k (row in mu) in K, update mu prototype vector
    for k in range(K):
        # Find sum of all R's in column k, r_sum
        r_sum = np.sum(R[:,k])
        
        # Re-initialize rx_sum
        rx_sum = 0

        # For each n (row in X) in N
        for n in range(N):
            # Find sum of r_nk * x_n, rx_sum
            rx_sum = rx_sum + R[n,k]*X[n]        
        # End for n

        # Set row k in mu_new as rx_sum/r_sum
        mu_new[k] = rx_sum/r_sum
        # Check for division by zero, if zero, reinitalize mu to random datapoint
        if r_sum == 0:
            mu_new[k] = X[np.random.randint(0,N)]
            # raise ZeroDivisionError("Division by zero in update_mu")
    # End for k

    # Return mu_new
    return mu_new

def k_means(X: np.ndarray, K: int, num_its: int) -> Union[list, np.ndarray, np.ndarray]:
    '''
    Performs the K-means algorithm according to the github instructions.

    Inputs:
    X       : A [N x f] array of samples.
    K       : Numberof clusters of data points
    num_its : The number of iterations

    Returns:
    J_hats  : List, length num_its, of the value of the objective function for each iteration
    Mu      : A [K x f] array of prototypes where each row is a prototype vector
    R       : A [N x K] array of indicators.
    '''
    # Standardize the sample data, find mean and standard deviation
    # Note: run the k_means algorithm on X_standard, not X.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std

    # Step 1: Initialize K prototypes by picking K random samples from X as prototypes, Mu
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: K], :]
    
    # Initialize J_hat as an empty list
    J_hats = []
    
    # For each iteration
    for iter in range(num_its):
    
        # Step 2: E-step, calculate distance matrix, dist_matrix, as well as determining R [N x K] matrix with each r_nk indicator
        dist_matrix = distance_matrix(X_standard, Mu)
        R = determine_r(dist_matrix)
        
        # Step 3: Calculate value of J hat, objective function, for all r_nk and mu_k. Append value to the list
        J_hats.append(determine_j(R, dist_matrix))        

        # Step 4: M-step, find new mus
        Mu = update_Mu(Mu, X_standard, R)
        
        # Step 5: Compare new J_hat to previous J_hat. If the difference is above a certain threshold, we perform steps 2-4 again. Otherwise we continue up to a maximum number of iterations.
        # Note: We have no threshold so we perform the number of iterations
    # End for iter

    # "de-standardize" the prototypes
        for i in range(K):
            Mu[i, :] = Mu[i, :] * X_std + X_mean

    # Return Mu, R and J_hats
    return [J_hats, Mu, R]

def _plot_j(iters: int, js: list):
    '''
    Plots the objective function as a function of iterations.
    Saves plot to 1_6_1.png

    inputs:
    iters   : Number of iterations
    js      : List of js corresponding to the iterations.
    '''
    # Plot js as a function of iters
    plt.plot(range(iters), js)    
    # Label axis
    plt.xlabel("Iterations")
    plt.ylabel("Objective function, J hat")
    # Label title
    plt.title("Running values of J hat ", fontsize = 20)
    
    # Save plot as 1_6_1.png
    plt.savefig("06a_kmeans\\1_6_1.png")

def _plot_multi_j(X: np.ndarray, iters: int = 10, ks: list = [2, 3, 5, 10]):
    '''
    Plots _plot_j()  times for each 
    Saves the plot to 1_7_1.png
    
    inputs:
    iters   : Number of iterations
    ks      : List of ks where each k is run in the k_means algorithm
    '''
    # Make list of j_hat lists
    j_hats = []
    
    # get number of ks
    n_ks = len(ks)
    
    # Find number of rows and number of columns
    nrows = int(np.sqrt(n_ks))
    ncols = nrows
    if nrows*ncols < n_ks:
        nrows += 1
            
    # Create subplots with shared x axis but not y axis
    fig, subplots = plt.subplots(nrows, ncols, sharex=True, sharey=False)
    # Title plot
    fig.suptitle("J hat as a function of iterations for different values of k")
    
    # Collect j_hats
    for k in range(n_ks):
        # Run k_means to collect j_hats
        j_hats.append(k_means(X,ks[k], iters)[0])
        
    for row in range(nrows):
        for col in range(ncols):
            k = row*ncols+col
            # Plot on subplot
            subplots[row][col].plot(range(iters),j_hats[k], label=("k = " + str(k)))
            # Label axis
            if row == (nrows-1):
                subplots[row][col].set_xlabel("Iterations")
            if col == 0:
                subplots[row][col].set_ylabel("J hat")
            # Title plot
            subplots[row][col].set_title("k = " + str(ks[k]))
    
    # Save plot as 1_7_1.png
    plt.savefig("06a_kmeans\\1_7_1.png")

def k_means_predict(X: np.ndarray, t: np.ndarray, classes: list, num_its: int) -> np.ndarray:
    '''
    Performs classification predictions for each datapoint in X using the k_means() algorithm.

    Inputs:
    X : A [N x f] array of input features
    t : A [N] array of target labels
    classes : A list of possible target labels. The number of classes, K, tells us how many clusters to make
    num_its : Number of k_means iterations

    Output:
    the predictions : [N] 
    '''
    # Find K, the number of classes
    K = len(classes)
    # Find N, the number of datapoints
    N = X.shape[0]
    
    # Run k_means algorithm to get J_hats, Mu and R
    J_hats, Mu, R = k_means(X, K, num_its)
    
    # Initialize predictions
    predictions = np.zeros(N, dtype=int)
    
    # Loop through each datapoint
    for n in range(N):
        # Reinitialize class_ID
        class_ID = -1

        # Loop through each column of R
        for k in range(K):
            # The column of R which is equal to 1 indicates which class the datapoint belongs to, get class_ID
            if R[n][k] == 1:
                class_ID = k
        # End for k
        # save the class ID to our predictions as integers
        predictions[n] = class_ID
    # End for n
    
    # return predictions
    return predictions

def _iris_kmeans_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float :
    '''
    Determine the accuracy of predictions made by k_means on a dataset [X, t] where we assume
    the most common cluster for a class label corresponds to that label.
    Accuracy = (Number of correct predictions) / number of data points
    
    inputs:
    targets     : The actual target classes
    predictions : The predicted classes
    
    output:
    accuracy    : The accuracy of the predictions
    '''
    # Find number of data points, N
    N = targets.shape[0]

    # Initialize number of correct predictions, N_true
    N_true = 0
    
    # Loop through each target, count number correct predictions, N_true
    for n in range(N):
        if predictions[n] == targets[n]:
            N_true += 1
            
    accuracy = N_true/N
    
    # Return accuracy
    return accuracy
    
def _iris_kmeans_confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''
    Determine the confusion matrix of predictions made by k_means on a dataset [X, t] where we
    assume the most common cluster for a class label corresponds to that label.
    
    inputs:
    targets     : The actual target classes
    predictions : The predicted classes
    
    output:
    confusion_matrix : [K x K] A matrix which tells us how many of each prediction is correct and how many of the incorrect ones belong to which class
                        The rows correspond to the target class and the columns to the predicted class.
    '''
    # Find out how many classes there are by getting the maximum value from either targets or predictions
    K = max(np.max(targets),np.max(predictions)) + 1
    
    # Initialize zero confusion matrix
    confusion_matrix = np.zeros([K, K])
    
    # Find number of data points, N
    N = len(targets)
    
    # Go through each prediction and target
    for n in range(N):
        # Increment the corresponding cell in the confusion matrix
        confusion_matrix[int(targets[n]),predictions[n]] += 1

    # Return confusion_matrix
    return confusion_matrix

def _my_kmeans_on_image(K, iters, ):
    '''
    I don't know what this function is for
    - Huldar, 2024-10-28
    '''
    # k_means()
    
    pass

def plot_image_clusters(n_clusters: list, image: np.ndarray, w: int, h: int):
    '''
    Plot the clusters found using sklearn k-means.
    
    inputs:
    n_clusters  : List with how many clusters to try
    image       : [N x 3] array with the RGB values of the image
    w           : Width of the image
    h           : Height of the image
    '''
    for k in range(len(n_clusters)):
        # image, (w, h) = image_to_numpy() # This was here when I got here, I don't agree with loading this data inside the function, better to pass it in - Huldar 2024-10-28
        kmeans = KMeans(n_clusters[k]).fit(image)

        plt.figure()
        plt.subplot(121)
        plt.imshow(image.reshape(w, h, 3))
        plt.subplot(122)
        plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
        # plt.show()
        # Save plot as 2_1_i.png where i is the number of this image
        plt.savefig("06a_kmeans\\2_1_" + str(k+1) + ".png")


def _gmm_info():
    ...


def _plot_gmm():
    ...
    
if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    print(" ")
    print("\nRunning template.py for assignment 6a")
    # Init grade estimate counter
    n_sections:int = 0
    n_sections_correct:int = 0
    
    '''
    K-means algorithm
    1. Initialize the k prototypes. Typically these are chosen at random from the set of samples.
    2. E-step:
        - Calculate the distance matrix ( D ) of size [ n × k ] . Element D [ i , j ] is the euclidean distance between sample i and prototype j .
        - Determine r n k for each point x n . We set r n k = 1 and r n j = 0 , j ≠ k if we decided x n belongs to k . Typically we make this determination by choosing the μ k closest to x n .
    3. We then calculate the value of J , our objective function: $$ J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} ||x_n - \mu_k ||^2 $$ our goal is to find values for all r n k and all μ k (our parameters) to minimize the value of J . Here, we will be using the average distance from the points to their cluster means as the objective value (let's call it J ^ ). $$ \hat{J} = \frac{1}{N}\sum_{n=1}^N \sum_{k=1}^K r_{nk} || x_n - \mu_k || $$
    4. M-step We now recompute the value of the prototypes: $$ \mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}} $$
    5. Compare the current value of J ^ to the previous value of J ^ . If the difference is above a certain threshold, we perform steps 2-4 again. Otherwise we continue up to a maximum number of iterations.
    '''
    # 1.1
    n_sections = n_sections+1
    print("1.1 - ",end="")
    
    a = np.array([[1, 0, 0],
                  [4, 4, 4],
                  [2, 2, 2]])
    b = np.array([[0, 0, 0],
                  [4, 4, 4]])
    dist_ab = distance_matrix(a, b)
    if str(dist_ab) == str(np.array([[1. ,6.40312424],[6.92820323, 0.],[3.46410162, 3.46410162]])):
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.2
    n_sections = n_sections+1
    print("1.2 - ",end="")
    dist = np.array([[  1,   2,   3],
                    [0.3, 0.1, 0.2],
                    [  7,  18,   2],
                    [  2, 0.5,   7]])
    r = determine_r(dist)

    if str(r) == str(np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0]])) :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.3
    n_sections = n_sections+1
    print("1.3 - ",end="")
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    r = determine_r(dist)
    j = determine_j(r, dist)
    
    if j == 0.9 :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.4
    n_sections = n_sections+1
    print("1.4 - ",end="")
    
    X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])
    Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])
    R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])
    
    mu_new = update_Mu(Mu, X, R)
    
    if str(mu_new) == str(np.array([[0.0,0.5,0.0],[1.0,0.0,0.0]])) :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.5
    n_sections = n_sections+1
    print("1.5 - ",end="")

    X, y, c = load_iris()
    K = 4
    num_its = 10
    J_hats, Mu, R = k_means(X, K, num_its)

    print("Complete")
    n_sections_correct = n_sections_correct + 0.5

    print("Mu:")
    print(Mu[1:10,:])
    print(".......")
    print("R:")
    print(R[1:10,:])
    print(".......")
    print("J_hats")
    print(J_hats[1:min(len(J_hats),10)])
    print(".......")
    
    print("Example output:")
    print('''array([ \n
        [6.15833333, 2.8875    , 4.82916667, 1.62916667],
        [5.50833333, 2.47083333, 3.82916667, 1.17083333],
        ...),
        array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        ...),
        [0.9844512801797404, 0.7868265220144953, ...]''')
    
    # 1.6
    n_sections = n_sections+1
    print("1.6 - ",end="")
    _plot_j(num_its, J_hats)
    print("Plot saved")
    n_sections_correct = n_sections_correct + 0.5

    # 1.7
    n_sections = n_sections+1
    print("1.7 - ",end="")
    ks = [2, 3, 5, 10]
    _plot_multi_j(X, num_its, ks)
    print("Plot saved")
    n_sections_correct = n_sections_correct + 0.5

    # 1.8
    n_sections = n_sections+1
    print("1.8 - ",end="")
    '''Answer the following questions:

    According to the results from section 1.7, what is the best value of k according to the objective function?
    If we have n samples and we set k = n , what would happen?
    Is it a good strategy to set k = n ? Why/Why not?'''
    # Make text answer
    text_answer = "1: 10, the highest value we tried.\n\
2: if K = N then each datapoint would be its own cluster and this whole thing would be kind of pointless.\n\
    Each datapoint would belong to its own \"cluster\" and the r matrix would reflect that resulting in no changes per iteration.\n\
3: It is not a good strategy to set K = N, see answer to question 1_8_2 for explanation."

    with open('.\\06a_kmeans\\1_8.txt', 'w') as f:
        f.write(str(text_answer))
    print("File updated")
    n_sections_correct = n_sections_correct + 0.5

    # 1.9
    n_sections = n_sections+1
    print("1.9 - ",end="")
    X, t, c = load_iris()
    num_its = 5
    predictions = k_means_predict(X, t, c, num_its)
    print("Complete")
    n_sections_correct = n_sections_correct + 0.5

    print("Predictions:")
    print(predictions)
    print("Example output:")
    print("[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n\
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n\
 0. 0. 2. 2. 2. 1. 1. 1. 2. 1. 1. 1. 1. 1. 1. 1. 1. 2. 1. 1. 1. 1. 1. 1.\n\
 ...\n\
 2. 1. 2. 1. 2. 2. 1. 1. 2. 2. 2. 2. 2. 1. 1. 2. 2. 2. 1. 2. 2. 2. 1. 2.\n\
 2. 2. 1. 2. 2. 1.]")

    # 1.10
    n_sections = n_sections+1
    print("1.10 - ",end="")
    '''Using the actual class labels of the samples and the predictions made by k_means determine the accuracy
    and confusion matrix of the prediction. You can use _iris_kmeans_accuracy for this.'''
    # Make text answer
    accuracy = _iris_kmeans_accuracy(t, predictions)
    confusion_matrix = _iris_kmeans_confusion_matrix(t, predictions)
    text_answer = "Accuracy:\n{:.03f}".format(accuracy) + "\nConfusion matrix:\n" + str(confusion_matrix)

    with open('.\\06a_kmeans\\1_10.txt', 'w') as f:
        f.write(str(text_answer))
    print("File updated")
    n_sections_correct = n_sections_correct + 0.5

    ## Section 2 - Clustering an image
    # 2.1
    n_sections = n_sections+1
    print("2.1")
    K = 7
    iters = 5
    
    # Get image data as X
    X_flattened, (w, h) = image_to_numpy("06a_kmeans/images/buoys.png")
        
    # J_hats, Mu, R = k_means(X_flattened, K, iters) # Took 3 minutes to run. Huldar 2024-10-28 18:43
    # _plot_j(iters, J_hats)
    num_clusters = [2, 5, 10, 20]
    plot_image_clusters(num_clusters, X_flattened, w, h)
    print("Plots saved")
    n_sections_correct = n_sections_correct + 0.5
    
    print("\n---------------------------------------------------------------\nRun succesful :)\n")
    print(("Estimated " + str(n_sections_correct) + " points out of " + str(n_sections)))
    print("Grade estimate: {:.2f}\n".format(10*n_sections_correct/n_sections))

'''
    if str() == str() :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")
'''