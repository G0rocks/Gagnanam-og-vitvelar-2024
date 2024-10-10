# Author: Huldar
# Date: 2024-09-25
# Project: Assignment 4
# Acknowledgements: 
#

# Imports
from typing import Union
import torch

from tools import load_iris, split_train_test

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    # Check if x<-100 and return 0.0 otherwise return sigmoid function of x
    if x.any() < -100:
        return 0.0

    return 1/(1+torch.exp(-x))

def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    Returns a tensor of the same size.
    '''
    return sigmoid(x)*(1-sigmoid(x))

def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Is kind of the activation function of each node ;)
    
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    # Make sure the sizes are the same, if not return None
    #if x.shape[1] != w.shape[0]:
    #    return None
    # Get weighed sum (dot product) of x and w
    w_sum = torch.matmul(x,w)
    # Return weighed sum and sigmoid of weighed sum
    return (w_sum, sigmoid(w_sum))

def _add_bias(x: torch.Tensor, bias=1) -> torch.Tensor:
    '''
    Adds a bias (default value of 1) as the first value of the input tensor x.
    Size of x is 1 x D
    Output [1, x0, x1, ..., xn] torch tensor
    '''
    # Find D
    D = x.size()[0]
    
    # Init empty tensor
    z = torch.empty((D+1))
    # Set bias
    z[0] = bias
    # Add inptu vector
    for i in range(D):
        z[i+1] = x[i]
    
    # Return tensor with added bias
    return z    
    
def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    ffnn = feed forward neural network
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    
    Inputs:
    x   : Input data point of size (1 x D) with D dimensions, a line vector
    M   : Number of hidden layer neurons
    K   : Number of output neurons
    W1  : Is a ((D + 1) x M) matrix. The +1 is for the bias weights. Represents the linear transform from the input layer to the hidden layer 
    W2  : Is a ((M + 1) x K) matrix. The +1 is for the bias weights. Represents the linear transform from the hidden layer to the output layer.
    
    Output:
    y   : The output of the neural network. Size (1 x K)
    z0  : The input pattern of size 1 x (D + 1)) , (this is just x with 1.0 inserted at the beginning to match the bias weight).
    z1  : The output vector of the hidden layer of size (1 x (M + 1)) (needed for backprop).
    a1  : The input vector of the hidden layer of size (1 x M) (needed for backprop).
    a2  : The input vector of the output layer of size (1 x K) (needed for backprop).
    '''
    # Go through the hidden layer
    # Generate z0 = bias + x
    x_biased = _add_bias(x)
    z0 = x_biased
    
    # Generate a1 and z1 (activation function without bias)
    (a1, z1_no_bias) = perceptron(x_biased, W1)
    #a1  = output[0]
    #z1_no_bias = output[1]
    
    # Add bias to z1
    z1 = _add_bias(z1_no_bias)

    # Generate a2 and y (activation function without bias)
    (a2, y) = perceptron(z1, W2)

    # Return outputs
    return y, z0, z1, a1, a2

def cross_entropy_error(y: torch.Tensor, t: torch.Tensor) -> float:
    '''
    Returns the cross entropy error for classification of this specific data point.
    Equation 5.80 and 6.36 from Bishop
    
    Inputs:
    y   : The neural networks estimate (the guess) for targets. Size K where K is the number of classes
    t   : The actual targets in one-hot notation. Size K where K is the number of classes
    
    Output:
    E  : The cross entropy error between y and t
    '''
    # Find N and K
    K = y.size(0)
    
    # Init dE
    E = torch.zeros(K)
    # For each possible class subtract target times logarithm of estimate
    for k in range(K):
        E[k] = - (t[k]*torch.log(y[k]))
    
    return E

def make_dE2(target: torch.Tensor, z1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    '''
    Makes the differentiated error matrix between the hidden layer and the output layer.
    Tells us how much of an error for this specific data point comes from which weight in W2.
    
    Inputs:
    target  : Size (1 X K) the correct output vector
    z1      : Size (1 X (M+1)) is the output of the hidden layer.
    W2      : Size ((M+1) X K) is the weight matrix between the hidden and output layers
    
    Note:
    K is the number of output neurons
    M is the number of hidden layer neurons
    
    Outpu:
    dE2 : Size ((M+1) X K) Gradient error matrix. Tells us how much of the total error comes from each weight in W2
    '''
    # Find K and M
    M_plus_1 = W2.size()[0]
    K = W2.size()[1]

    # Initialize dE2
    dE2 = torch.zeros((M_plus_1, K))
    
    print("Target: " + str(target))
    # Loop through each row and column of dE2 and fill value
    for m in range(M_plus_1):   # Loop through rows
        for k in range(K):      # Loop through columns
            a = z1[m]*W2[m][k]
            dE2[m][k] = target[k].item()*d_sigmoid(a)/sigmoid(a)
        print("Last column value: " + str(a))
    
    # Return dE2
    return dE2

def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y.
    The point being to see how far away from the target_y we are in the forward propagation
    and return how much of the error can be attributed to each layer so that in another
    function we can use the error to adjust each weight.
    
    Note: Only backpropagates once for one data point x, not the whole dataset.
    Assumes the sigmoid function is the hidden and output activation function
    Assumes cross-entropy error function (for classification).
    Notice that E_n(w) (where w is the weight vector) is defined as the error function for a single data point x_n.
    The algorithm is described on page 237 in Bishop (page 244 in old Bishop).

    Inputs:
    x           : Input data point of size (1 x D) with D dimensions, a line vector
    target_y    : The target vector in 1 hot encoding, size (1 x K) where all values are 0 except one, which is 1.
    M           : Number of hidden layer neurons
    K           : Number of output neurons
    W1          : Is a ((D + 1) x M) matrix. The +1 is for the bias weights. Represents the linear transform from the input layer to the hidden layer 
    W2          : Is a ((M + 1) x K) matrix. The +1 is for the bias weights. Represents the linear transform from the hidden layer to the output layer.

    Outputs:
    y   : Is the output of the output layer, size (1 x K)
    dE1 : The gradient error matrices that contain (delta E / delta w_ji) for the first layer.
    dE2 : The gradient error matrices that contain (delta E / delta w_ji) for the second layers.
    '''
    # Step 1: Forward propagation A.K.A. run ffnn on the input.
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    # Step 2: Calculate the output error delta_k = y_k − target_y_k
    delta_k = y-target_y
    
    # Step 3: Calculate: the hidden layer error delta_j = \frac{d}{da} sigma(a_j^1)*sum_k(w_{k, j+1}*delta_k) (the +1 is because of the bias weights)
    # delta_j_Huldar = d_sigmoid(a1)*torch.sum(torch.matmul(W2, delta_k))
    delta_j_Elisa = d_sigmoid(a1) * torch.matmul(W2[1:, :], delta_k)
    # Currently using Elisas delta j formula since she got this section correct and I did not.
    delta_j = delta_j_Elisa
 
    # Step 4: Initialize dE1 and dE1 as zero-matrices with the same shape as W1 and W2
    # Find Error using cross-entropy error function (for classification), eq. 5.80 and eq. 6.36 in Bishop
    dE1 = torch.zeros(W1.size())
    dE2 = torch.zeros(W2.size())
    
    E = cross_entropy_error(y, target_y)
    dE2 = make_dE2(target_y, z1, W2)
        
    # Step 5: Calculate dE1_{i,j} = delta_j*z_i^(0) and dE2_{j,k} = delta_k*z_j^(1)
    # dE1 = torch.matmul(delta_j,z0)
    # dE2 = torch.matmul(delta_k, z1)
    # (delta E / delta w_ji)
    # dE2 = dE / delta_k
    # dE1 = torch.matmul(d_sigmoid(a1),torch.matmul(dE2,torch.transpose(W2,0,1)))
    
    # Step 5 from Elisa - Update gradients for dE1 and dE2
    # For the first layer weights (between input and hidden layer)
    for i in range(W1.shape[0]):  # Loop over D+1 (input + bias)
        for j in range(W1.shape[1]):  # Loop over M (hidden neurons)
            dE1[i, j] = delta_j[j] * z0[i]
    
    # For the second layer weights (between hidden and output layer)
    for j in range(W2.shape[0]):  # Loop over M+1 (hidden neurons + bias)
        for k in range(W2.shape[1]):  # Loop over K (output neurons)
            dE2[j, k] = delta_k[k] * z1[j]
    
    # Return outputs
    return y, dE1, dE2

def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    
    Inputs:
    X_train     : Training data, size (N X D)
    t_train     : Training data target values [N]
    M, K, W1, W2 are defined as above
    iterations  : Number of iterations the training should take, i.e. how often we should update the weights
    eta         : Learning rate.

    
    Outputs:
    W1tr                    : Updated weight matrix for left side of neural network. Size ()
    W2tr                    : Updated weight matrix for right side of neural netwprk. Size ()
    E_tot                   : 
    misclassification_rate  : 
    guesses                 :
    '''
    pass
    # Return  W1tr, W2tr, E_total, misclassification_rate, guesses
    return  W1tr, W2tr, E_total, misclassification_rate, guesses


def test_nn(
    X: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> torch.Tensor:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # Init grade estimate counter
    n_sections = 0
    n_sections_correct = 0
    
    # 1
    # 1.1 - Sigmoid function
    n_sections = n_sections+1
    print("\n1.1 - Sigmoid function")
    if str(sigmoid(torch.Tensor([0.5]))) == str(torch.tensor([0.6225])) and str(d_sigmoid(torch.Tensor([0.2]))) == str(torch.tensor([0.2475])):
        print("Pass")
        n_sections_correct = n_sections_correct+1
    else:
        print("Fail")

    # 1.2 - Perceptron function
    n_sections = n_sections+1
    print("1.2 - Perceptron function")
    
    if str(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1]))) == str((torch.tensor(1.0800), torch.tensor(0.7465))) and str(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4]))) == str((torch.tensor(0.1800), torch.tensor(0.5449))):
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")
    
    # 1.3 - Forward propagation
    print("1.3 - Forward propagation")
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    data, targets, classes = load_iris()
    (train_data, train_targets), (test_data, test_targets) = split_train_test(data, targets)
    
    # initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point:
    x = train_data[0, :]
    K = 3  # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    # Check results
    if  str(y) == str(torch.tensor([0.7079, 0.7418, 0.2414])) and \
        str(z0) == str(torch.tensor([1.0000, 4.8000, 3.4000, 1.6000, 0.2000])) and \
        str(z1) == str(torch.tensor([1.0000, 0.9510, 0.1610, 0.8073, 0.9600, 0.5419, 0.9879, 0.0967, 0.7041, 0.7999, 0.1531])) and \
        str(a1) == str(torch.tensor([ 2.9661, -1.6510,  1.4329,  3.1787,  0.1681,  4.4007, -2.2343,  0.8669, 1.3854, -1.7101])) and \
        str(a2) == str(torch.tensor([ 0.8851,  1.0554, -1.1449])):

            print("Pass")
            n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")

    # 1.4 - Backward propagation
    n_sections = n_sections+1
    print("1.4 - Backward propagation")
    # initialize random generator to get predictable results
    torch.manual_seed(42)

    K = 3  # number of classes
    M = 6
    D = train_data.shape[1]
    print("Dimensions: " + str(D))
    x = data[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

    if str(y) == str(torch.tensor([0.7047, 0.5229, 0.5533])):
        print("y correct")
        n_sections_correct = n_sections_correct + 1/3
    else:
        print("y incorrect")
        
    # print(str(dE1))
    # print(str(torch.tensor([[ 3.0727e-02,  4.4601e-03,  4.3493e-04,  1.7125e-02,  9.1134e-04, -7.7194e-02],
    #                         [ 1.5671e-01,  2.2747e-02,  2.2181e-03,  8.7336e-02,  4.6478e-03, -3.9369e-01],
    #                         [ 1.0755e-01,  1.5611e-02,  1.5222e-03,  5.9936e-02,  3.1897e-03, -2.7018e-01],
    #                         [ 4.3018e-02,  6.2442e-03,  6.0890e-04,  2.3974e-02,  1.2759e-03, -1.0807e-01],
    #                         [ 6.1455e-03,  8.9203e-04,  8.6985e-05,  3.4249e-03,  1.8227e-04, -1.5439e-02]])))

    '''
    # print(str(dE2))
    # print(str(torch.tensor([[-0.2953,  0.5229,  0.5533],
                                [-0.1518,  0.2687,  0.2844],
                                [-0.2923,  0.5175,  0.5476],
                                [-0.2938,  0.5201,  0.5504],
                                [-0.0078,  0.0139,  0.0147],
                                [-0.2948,  0.5220,  0.5524],
                                [-0.2709,  0.4796,  0.5075]])))
    '''
    
    if  str(dE1) == str(torch.tensor([[ 3.0727e-02,  4.4601e-03,  4.3493e-04,  1.7125e-02,  9.1134e-04, -7.7194e-02],
                                        [ 1.5671e-01,  2.2747e-02,  2.2181e-03,  8.7336e-02,  4.6478e-03, -3.9369e-01],
                                        [ 1.0755e-01,  1.5611e-02,  1.5222e-03,  5.9936e-02,  3.1897e-03, -2.7018e-01],
                                        [ 4.3018e-02,  6.2442e-03,  6.0890e-04,  2.3974e-02,  1.2759e-03, -1.0807e-01],
                                        [ 6.1455e-03,  8.9203e-04,  8.6985e-05,  3.4249e-03,  1.8227e-04, -1.5439e-02]])) and \
        str(dE2) == str(torch.tensor([[-0.2953,  0.5229,  0.5533],
                                        [-0.1518,  0.2687,  0.2844],
                                        [-0.2923,  0.5175,  0.5476],
                                        [-0.2938,  0.5201,  0.5504],
                                        [-0.0078,  0.0139,  0.0147],
                                        [-0.2948,  0.5220,  0.5524],
                                        [-0.2709,  0.4796,  0.5075]])):
        
        print("Pass")
        n_sections_correct = n_sections_correct + 2/3
    else:
        print("Fail")
    
    # 2 - Training the network
    # 2.1
    n_sections = n_sections+1
    print("2.1")
    W1tr, W2tr, E_total, misclassification_rate, guesses = train_nn(X_train, t_train, M, K, W1, W2, iterations, eta)
    
    
    
    
    # 2.2
    n_sections = n_sections+1
    print("2.2")
    
    # 2.3
    n_sections = n_sections+1
    print("2.3")
    

    
    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")
    print(("Estimated {:.2f} points out of " + str(n_sections) + "\nGrade estimate: {:.2f}\n").format(n_sections_correct, 10*n_sections_correct/n_sections))

'''
    if str() == str() :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")
'''