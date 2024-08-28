# Author: Huldar
# Date: 2024-08-28
# Project: Assignment 1
# Acknowledgements: 
#

import matplotlib.pyplot as plt
import numpy as np

from tools import scatter_2d_data, bar_per_axis

RANDOM_SEED = 1234

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    '''
    Generate n values samples from the k-variate
    normal distribution.
    n: Number of vectors
    k: Dimension of each vector
    mean:   Mean for K-variate normal distribution
    var:    variance for K-variate normal distribution

    Returns nxK array X of vectors x_i
    '''    
    # Create variance array (no covariance)
    variance_matrix = np.identity(k)*var*var

    # Reset random seed
    np.random.seed(RANDOM_SEED)

    X_array = np.random.multivariate_normal(mean, variance_matrix, size=n)

    # Return X_array
    return X_array

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''
    Performs the mean sequence estimation update.
    mu: Mean value of old data
    x:  Array with new data that we want to add to the updated mean
    n:  Number of values in the x array + number of values in the mean already

    Loops through each value and estimates the new mean using the mean sequence estimation update method where
    
    mu_new = mu_old + (x[N]-mu_old)/N
    '''
    print("Update sequence mean")
    # Number of values in x
    n_in_x = x.shape[0]
   
    # Find how many values are already in the mean
    n_old = n - n_in_x
    # print("N old: " + str(n_old))
    # print("N in x: " + str(n_in_x))
    
    # mean as mu_old
    mu_old = mean
    # For each x value, update estimate for mean
    for N in range(n_in_x):
        # Find new mu
        mu_new = mu_old + (x[N] - mu_old)/(n_old + N)
        # update old mu
        mu_old = mu_new
    
    return mu_new

def _plot_sequence_estimate(data, save_fig=None):
    '''
    Plots a sequence estimate for all the vectors in the data with an initial estimate of zero with k-dimensions
    data:   n x k array with an n number of k-dimensional vectors
    save_fig:   "Name of file to save figure into, if no file name is given, will show plot and not save. Default value is None.
    '''
    # Initialize inital estimate as zero array with k-dimensions
    k = data.shape[1]
    estimates = [np.zeros(k)]   # Note, this is a list of mean estimation vectors
    
    '''
    From template
        data = None # Set this as the data
    estimates = [np.array([0, 0])]
    '''

    # For each vector in the data matrix, get an updated_sequence_mean. Collect estimates as we go
    for i in range(data.shape[0]):
        # Add new estimate to collection
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))
        
    '''
    From template
        for i in range(data.shape[0]):
        """
            your code here
        """
    '''

    # Generate plot from estimation for each dimension
    for i in range(k):
        plt.plot([e[i] for e in estimates], label=(str(i+1)+' dimension'))

    '''
    From template:
        plt.plot([e[0] for e in estimates], label='First dimension')
    """
        your code here
    """
    '''

    # Title plot
    plt.title("Mean estimates per data point")
    # Label axis
    plt.xlabel("Data points used for estimation") # Add ", fontsize = #" to control fontsize
    plt.ylabel("Mean estimation")
    plt.legend(loc='upper center')
    
    if save_fig == None:
        # Show plot
        plt.show()
        return
    
    plt.savefig(save_fig)

def _square_error(y, y_hat):
    '''
    Finds the square error between estimation and actual mean after every update.
    y:      Actual mean
    y_hat:  Estimation

    returns error averaged across all dimensions (average error)
    '''
    # Calculate error for each dimension
    error = np.power(y-y_hat,2)
    # Return average error
    return np.average(error)

def _plot_mean_square_error(data, mean, save_fig=None):
    '''
    Finds the square error between estimation and actual mean after every update.
    Initial estimation is a zero vector.
    data:       n x k array with an n number of k-dimensional vectors
    mean:       k-dimensional mean vector
    save_fig:   "Name of file to save figure into, if no file name is given, will show plot and not save. Default value is None.
    '''
    # Initialize inital estimate as zero array with k-dimensions
    k = data.shape[1]   # Dimensions
    estimates = [np.zeros(k)]   # Note, this is a list of mean estimation vectors

    # For each vector in the data matrix, get an updated_sequence_mean. Collect estimates as we go
    for i in range(data.shape[0]):
        # Add new estimate to collection
        estimates.append(update_sequence_mean(estimates[i], data[i], i+1))

    # Initialize empty error_list
    error_list = []

    # For each estimation, find error between true mean and estimated mean
    for estimate in estimates:
        error_list.append(_square_error(mean, estimate))

    # Clear figure
    plt.clf()

    # Plot error over each estimate:
    plt.plot(error_list)
    
    # Title plot
    plt.title("Average error between estimate and mean per data point")
    # Label axis
    plt.xlabel("Data points used for estimation") # Add ", fontsize = #" to control fontsize
    plt.ylabel("Error")

    # Plot show/save
    if save_fig == None:
        # Show plot
        plt.show()
        return
    
    plt.savefig(save_fig)


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: np.float64
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass



# Test area
# -----------------------------------------------------
if __name__ == '__main__':
    print("\nStarting test")
    # Part 1
    '''
    print("Part 1\n is:")
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    print("\n The same as\n")
    print(np.array([[ 0.61286571, -0.5482684 ,  0.86251906],
                    [-0.40644746,  0.06323465,  0.15331182]]))
    print("\n And is:\n")
    print(gen_data(5, 1, np.array([0.5]), 0.5))
    print("\n The same as\n")
    print(np.array([[ 0.73571758],
                    [-0.09548785],
                    [ 1.21635348],
                    [ 0.34367405],
                    [ 0.13970563]]))
    '''

    # Part 2
    print("Part 2")
    # Create 300 3-dimensional data points sampled, write to file
    n_points = 300
    vector_dimensions = 2
    mean = [-1,2]
    var = np.sqrt(4)
    data_2 = gen_data(n_points, vector_dimensions, mean, var)
    # Plot 2D data
    # scatter_2d_data(data_2)
    # bar_per_axis(data_2)

    # Answer to written question by writing to 2.txt
    print("Writing to 2.txt file")
    text_answer = "Do you expect the batch estimate to be exactly (0,1,-1)?\nNo, not exactly, but very close because that's what we generated it as.\n\nWhich two parameters can be used to make this estimate more accurate? I have no clue, really don't understand the statistics I'm doing right now but my feeling is that if we decrease the variance and increase the number of values then we'll get means that are closer to the input values.\nhttps://m.media-amazon.com/images/I/41sKf2ToyPL.jpg"
    with open('2.txt', 'w') as f:
        f.write(str(text_answer))
    print("File updated")


    # Part 3
    print("Part 3\nIs this:")
    mean = np.mean(data_2, 0)
    new_data = gen_data(1, 2, np.array([0, 0]), 1)
    # print("mean: " + str(mean))
    # print("New data: " + str(new_data))
    # print("N: " + str(data_2.shape[0]+new_data.shape[0]))
    print("Is this: " + str(update_sequence_mean(mean, new_data, data_2.shape[0]+1)))
    print("Close to this: [[-0.85286428  1.95485036]])?")

    # Part 4
    print("Part 4")
    X = data_2
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    print(update_sequence_mean(mean, new_x, X.shape[0]))


    # Part 5
    print("Part 5")
    # Create 100 3-dimensional data points with mean [0,0,0] and variance 4
    mean = [0,0,0]
    variance = 4
    data_5 = gen_data(100, 3, mean, variance)
    _plot_sequence_estimate(data_5, save_fig=".\\03_sequential_estimation\\5.png")
    #_plot_sequence_estimate(data_5)


    # Part 6
    print("Part 6")
    #_plot_mean_square_error(data_5, mean)
    _plot_mean_square_error(data_5, mean, ".\\03_sequential_estimation\\6_1.png")


    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''
