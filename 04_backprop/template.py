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
    if x < -100:
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
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    # Make sure the sizes are the same, if not return None
    if x.shape != w.shape:
        return None
    # Get weighed sum (dot product) of x and w
    w_sum = torch.dot(x,w)
    # Return weighed sum and sigmoid of weighed sum
    return (w_sum, sigmoid(w_sum))
    
def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    ...


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