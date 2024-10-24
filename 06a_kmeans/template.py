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
    r : A [N x K] array where r[i, j] is 1 if sample i is closest to prototype j and 0 otherwise.
    '''
    # Init r as zeros of size (N x K)
    N, K = dist.shape
    r = np.zeros([N,K],dtype=int)
    
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
        
        # Set value for r at index as 1
        r[n,index] = 1
    # end for n
            
    return r

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given arrays of indicators and distances.
    Based on equation 15.1 from Bishop

    Inputs:
    R : [N x K] array where R[i, j] is 1 if sample i is closest to prototype j and 0 otherwise.
    dist : [N x K] array of distances

    output:
    j : The value of the objective function
    '''
    # Get linear combination of R and dist for each row in dist
    # Find minimum value
    # j = np.min(dist.dot(R.transpose()))
    N, K = dist.shape
    
    j = 0
    
    for n in range(N):
        for k in range(K):
            j = j + r[n][k]*dist[n][k] #*dist[n][k]
    
    # return j
    return j/N

def update_Mu(Mu: np.ndarray, X: np.ndarray, R: np.ndarray) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current prototypes, samples and indicators.

    Inputs:
    Mu: A [K x f] array of current prototypes.
    X : A [N x f] array of samples.
    R : A [N x K] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    ...


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    ...

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    ...


def _plot_j():
    ...


def _plot_multi_j():
    ...


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    ...


def _iris_kmeans_accuracy():
    ...


def _my_kmeans_on_image():
    ...


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    ...
    plt.subplot('121')
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot('122')
    # uncomment the following line to run
    # plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


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
    
    if str(update_Mu(Mu, X, R)) == str(np.array([[0.0,0.5,0.0],[1.0,0.0,0.0]])) :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.5
    n_sections = n_sections+1
    print("1.5 - ",end="")

    X, y, c = load_iris()
    k_means(X, 4, 10)
    

    # 1.6
    n_sections = n_sections+1
    print("1.6 - ",end="")

    # 1.7
    n_sections = n_sections+1
    print("1.7 - ",end="")

    # 1.8
    n_sections = n_sections+1
    print("1.8 - ",end="")

    # 1.9
    n_sections = n_sections+1
    print("1.9 - ",end="")

    # 1.10
    n_sections = n_sections+1
    print("1.10 - ",end="")

    ## Section 2 - Clustering an image
    # 2.1
    n_sections = n_sections+1
    print("2.1")
   
    
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