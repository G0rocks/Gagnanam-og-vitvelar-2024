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
    * design_matrix - [NxM] is the basis function vectors containing a basis function
    output design_matrix for each data vector x in features
    '''
    # Get dimensions
    N, D = features.shape   # Get N, the number of data points and D the number of dimensions per data point
    M = mu.shape[0] # Get M, the number of rows in the mean tensor
    design_matrix = torch.zeros(N, M)  # Init design_matrix, basically it's a similarity score for how similar each data point is to each mean

    # Generate covariance matrix, Sigma_k???? Or is var the covariance matrix?
    cov_matrix = var*torch.eye(D)
    
    # Get multivariate normal, design_matrix
    # Use gaussian basis function formula to find design_matrix vectors
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
        
        # Compute the PDF for all features (NxD) and store it in the ith column of design_matrix
        design_matrix[:, i] = torch.tensor(mvn.pdf(features.numpy()))
    
    # design_matrix = multivariate_normal.pdf(features, mean_vec, cov_matrix) # NOT WORKING BECAUSE OF NUMPY ERROR!!!
    # design_matrix = torch.exp(-0.5 * torch.transpose((features-mu)) * cov_matrix^(-1) * (features - mu)) /(((2*torch.pi)^(D/2)) * (torch.abs(cov_matrix)^(0.5)))

    # Return phi
    return design_matrix

def _plot_mvn():
    '''
    Plot the output of each basis function, using the same parameters as above, as a function of the features.
    You should plot all the outputs onto the same plot.
    Turn in your plot as 2_1.png.
    '''
    # Loop through each basis function
    for i in range(M):
        # Plot basis function as a result of features, design_matrix output on Y-axis and feature ID on X-axis
        plt.plot(range(N), design_matrix[:,i],  label = "Bias function " + str(i+1))

    # Title
    plt.title("Basis functions - Output from each data point")
    
    # Legend
    # plt.legend()
    
    # Label axis
    plt.xlabel("Features - ID") # Add ", fontsize = #" to control fontsize
    plt.ylabel("design_matrix(x)")
    
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
    * fi: [NxM] is the array of basis function vectors - THIS IS THE DESIGN MATRIX FROM EQ. (4.15) IN BISHOP!!!
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant --> regularization matrix (lamda*I) [MxM]

    Output: [Mx1], the maximum likelihood estimate of w for the linear model, w_ML
    '''
    # We have the design matrix (I call it a basis function matrix) - Eq. (4.15) from Bishop
    design_matrix = fi
    
    # Find Moore-Penrose pseudo-inverse (MPp_inverse) of design matrix with regularization coefficient - Eq. (4.14) modified to Eq. (4.27) from Bishop
    # We can't use this since we have the lambda
    '''
    # If design_matrix is square and inversible then Moore-Penrose pseudo-inverse of design matrix is the inverse of design matrix - According to chapter 4.1.3 in Bishop
    if M == N:  # If design_matrix is square
        if torch.det(design_matrix) != 0:
            MPp_inverse = torch.inverse(design_matrix)
    else:
        design_matrix_T = torch.transpose(design_matrix, N, M)
        MPp_inverse = torch.inverse(design_matrix_T*design_matrix)*design_matrix_T
        
    # Finding maximum likelihood for weights (w_ML) - Eq. (4.14) from Bishop
    w_ML = MPp_inverse*targets
    '''
    
    # Init regularization matrix (I matrix multiplied by lambda) - Note: Dimension M x M because of eq. (4.27)
    reg_matrix = torch.eye(M)*lamda

    # Find transpose of design matrix (Swap dimension 0, rows, and dimension 1, columns)
    design_matrix_T = torch.transpose(design_matrix,0,1)
    
    # Finding maximum likelihood for weights (w_ML) - Eq. (4.27) from Bishop
    w_ML = torch.matmul(torch.inverse(reg_matrix + torch.matmul(design_matrix_T, design_matrix)), design_matrix_T)*targets
    
    # return weights
    return w_ML

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
    design_matrix = mvn_basis(X, mu, sigma)
    print("design_matrix type: " + str(type(design_matrix)) + "\ndesign_matrix shape: " + str(design_matrix.shape))

    # Part 2
    print("Part 2")
    print("Plotting...")
    _plot_mvn()
    print("Plot finished :)")

    # Part 3
    print("Part 3")
    # Estimate maximum likelihood - Linear regression
    lamda = 0.001
    w_ml = max_likelihood_linreg(design_matrix, t, lamda)
    print("Maximum likelihood: " + str(w_ml))

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