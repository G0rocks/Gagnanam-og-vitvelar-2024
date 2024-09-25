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
    z0 = _add_bias(x)
    
    # Generate a1 and z1 (activation function without bias)
    (a1, z1_no_bias) = perceptron(z0, W1)
    #a1  = output[0]
    #z1_no_bias = output[1]
    
    # Add bias to z1
    z1 = _add_bias(z1_no_bias)

    # Generate a2 and y (activation function without bias)
    (a2, y) = perceptron(z1, W2)

    # Return outputs
    return y, z0, z1, a1, a2


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
    for the given input pair x, target_y
    '''
    ...


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
    '''
    ...


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
    # 1
    # 1.1 - Sigmoid function
    print("\n1.1 - Sigmoid function")
    if str(sigmoid(torch.Tensor([0.5]))) == str(torch.tensor([0.6225])) and str(d_sigmoid(torch.Tensor([0.2]))) == str(torch.tensor([0.2475])):
        print("Pass")
    else:
        print("Fail")

    # 1.2 - Perceptron function
    print("1.2 - Perceptron function")
    
    if str(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1]))) == str((torch.tensor(1.0800), torch.tensor(0.7465))) and str(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4]))) == str((torch.tensor(0.1800), torch.tensor(0.5449))):
        print("Pass")
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
    else:
        print("Fail")

    # 1.4 - Backward propagation
    print("1.4 - Backward propagation")


    # 2 - Training the network
    # 2.1
    #print("2.1")
    
    
    # 2.2
    #print("2.2")
    
    # 2.3
    #print("2.3")
    

    
    
    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")

'''
    if == :
        print("Pass")
    else:
        print("Fail")
'''