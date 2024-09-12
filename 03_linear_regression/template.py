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
    var: float
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
    * var: All normal distributions are isotropic with sigma^2*I covariance
    matrices (where I is the MxM identity matrix)
    
    Note: The variable, var, is the variance, NOT THE COVARIANCE MATRIX!
    
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    # Get dimensions
    N, D = features.shape   # Get N, the number of data points and D the number of dimensions per data point
    M = mu.shape[0] # Get M, the number of rows in the mean tensor
    fi = torch.zeros(N, M)  # Init fi, basically it's a similarity score for how similar each data point is to each mean

    # Generate covariance matrix, Sigma_k???? Or is var the covariance matrix?
    cov_matrix = var*torch.eye(D)
    
    # Get multivariate normal, fi
    # Use gaussian basis function formula to find fi vectors
    # fi_k(x) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\Sigma_k|^{1/2}} e^{-\frac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)}
    # Note, can be called with the multivariate_normal() function from scipy
    # from torch.distributions.multivariate_normal import MultivariateNormal
    
    
    # Loop over each basis function mean vector - Loop made with assistant of ChatGPT, problematic since I have no idea how it works now or why
    for i in range(M):
        # Get the current mean vector
        mean_vec = mu[i].numpy()
        
        # Calculate the multivariate normal distribution for the current mean
        # and the isotropic covariance matrix
        mvn = multivariate_normal(mean=mean_vec, cov=cov_matrix)
        
        # Compute the PDF for all features (NxD) and store it in the ith column of fi
        fi[:, i] = torch.tensor(mvn.pdf(features.numpy()))
    
    # fi = multivariate_normal.pdf(features, mean_vec, cov_matrix) # NOT WORKING BECAUSE OF NUMPY ERROR!!!
    # fi = torch.exp(-0.5 * torch.transpose((features-mu)) * cov_matrix^(-1) * (features - mu)) /(((2*torch.pi)^(D/2)) * (torch.abs(cov_matrix)^(0.5)))

    # Return phi
    return fi


def _plot_mvn():
    '''
    Plot the output of each basis function, using the same parameters as above, as a function of the features.
    You should plot all the outputs onto the same plot.
    Turn in your plot as 2_1.png.
    '''
    # Loop through each basis function
    for i in range(M):
        # Plot basis function as a result of features, fi output on Y-axis and feature ID on X-axis
        plt.plot(range(N), fi[:,i],  label = "Bias function " + str(i+1))

    # Title
    plt.title("Basis functions - Output from each data point")
    
    # Legend
    # plt.legend()
    
    # Label axis
    plt.xlabel("Features - ID") # Add ", fontsize = #" to control fontsize
    plt.ylabel("fi(x)")
    
    # Save plot as 2_1.png
    #plt.show()
    plt.savefig(".\\Gagnanam-og-vitvelar-2024 git repo\\03_linear_regression\\2_1.png")
    


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
    var: float,
    w: torch.Tensor
) -> torch.Tensor:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * var: All normal distributions are isotropic with sigma^2*I covariance
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
    print(" ")
    print("\nPart 1")
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
    print("fi type: " + str(type(fi)) + "\nfi shape: " + str(fi.shape))

    # Part 2
    print("Part 2")
    print("Plotting...")
    _plot_mvn()
    print("Plot finished :)")

    # Part 3
    print("Part 3")
    # Estimate maximum likelihood - Linear regression
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print("Maximum likelihood: " + str(wml))

    # Part 4
    print("Part 4")


    # Part 5
    print("Part 5")



    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''