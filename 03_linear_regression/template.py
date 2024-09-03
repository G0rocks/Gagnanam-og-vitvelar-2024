# Author: Huldar
# Date: 2024-09-02
# Project: Assignment 3
# Acknowledgements: 
#

# Imports
import torch
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal

def mvn_basis(
    features: torch.Tensor,
    mu: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>

    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    # Get dimensions
    N, D = features.shape   # Get N, the number of data points and D the number of dimensions per data point
    M = mu.shape[0] # Get M, the number of rows in the mean tensor
    fi = torch.zeros(N, M)  # Init fi, the *INSERT WHAT FI IS HERE*

    # Generate covariance matrix
    covs = sigma*sigma*torch.eye(M)
    
    # Get multivariate normal, fi
    # Use gaussian basis function formula to find fi vectors
    # fi_k(x) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\Sigma_k|^{1/2}} e^{-\frac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)}
    # Note, can be called with the multivariate_normal() function from scipy
    fi = multivariate_normal(mu, covs)

    # Return phi
    return fi


def _plot_mvn():
    pass


def max_likelihood_linreg(
    fi: torch.Tensor,
    targets: torch.Tensor,
    lamda: float
) -> torch.Tensor:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    pass


def linear_model(
    features: torch.Tensor,
    mu: torch.Tensor,
    sigma: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    pass


# Test area
#---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    # Part 1
    print("Part 1")
    # Initialize data
    X, t = load_regression_iris()   # X = flower features, t = flower petal length
    # Get N, the number of data points and D the number of dimensions per data point
    N, D = X.shape

    # How many basis functions
    M = 10  # How many means per dimension we want
    sigma = 10  # Standard deviation
    mu = torch.zeros((M, D))    # Init mean
    # For each dimension, make M linearly spacesd from the smallest to the largest value of the dataset in that dimension
    for i in range(D):
        mmin = torch.min(X[:, i])
        mmax = torch.max(X[:, i])
        mu[:, i] = torch.linspace(mmin, mmax, M)
    # Generate fi (a.k.a. phi), result of the multivariate normal basis function
    fi = mvn_basis(X, mu, sigma)
    print(fi)



    # Part 1.2
    print("Part 2")

    # Part 1.3
    print("Part 1.3")


    # Part 1.4
    print("Part 1.4")


    # Part 1.5
    print("Part 1.5")



    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''