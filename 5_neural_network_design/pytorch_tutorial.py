'''
Authors: Danielle and Huldar
Date: 2024-10-09
Project:
Stuff for assignment 5 in data mining
Done from https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
'''
print("Running pytorch tutorial")

# Imports
# For tensors
import torch
import numpy as np
# For a gentle introduction to torch.autograd
from torchvision.models import resnet18, ResNet18_Weights
# For neural networks
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Training a classifier
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# For knowing the runtime
#import time

print("Imports complete")

#start_t = time.time()


# Tensors
#----------------------------------------
'''
print("Tutorial section 1, Tensors, start")
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")# We move our tensor to the GPU if available

# Indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# Tensor concatenation
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# This computes the tensor multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)

x = tensor
y = tensor
x.copy_(y)
x.t_()

# Bridge to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # Note only a memory reference, not a copy
print(f"n: {n}")

# Change tensor, see how numpy array reacts
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n) # Note only a memory reference, not a copy

# Change numpy array, see how tensor reacts
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

print("Tutorial section 1, Tensors, complete")
'''

#-----------------------------------------------------------------------
# A gentle introduction to torch.autograd

'''
print("Tutorial section 2, A gentle introduction to torch.autograd, start")

# Init resnet18 pretrained neural network, 18 layers deep
model = resnet18(weights=ResNet18_Weights.DEFAULT)
# Init random data, 1 by 3 by 64 by 64 (to represent random images)
data = torch.rand(1, 3, 64, 64)
# Init a 1 row by 1000 column data label tensor
labels = torch.rand(1, 1000)

# Make forward pass to make a prediction
prediction = model(data) # forward pass

# Calculate loss (error) from forward pass
loss = (prediction - labels).sum()
# Backpropagate error through the network.
loss.backward() # backward pass

# Load stochastic gradient descent (SGD) optimizer
learning_rate = 0.01
momentum = 0.9
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Initiate gradient descent
optim.step() #gradient descent

# Now let's take a look at differentiation in Autograd
# Let's make some tensors, requires_grad=True signals the autograd that every operation on them should be tracked
# Assume a and b are parameters of the neural network
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# Assume Q is the error function of the neural network
Q = 3*a**3 - b**2

# In NN training, we want gradients of the error with respect to the parameters
# In other words, we want the partial derivatives dQ/da and dQ/db
# We know that dQ/da = 9a^2
# We know that dQ/db = -2b


external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# Check the partial derivatives, if the collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

print("Tutorial section 2, A gentle introduction to torch.autograd, complete")
'''

#-----------------------------------------------------------------------
# Neural networks

'''
print("Tutorial section 3, Neural networks, start")

# Define a neural network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
print(net)

# Show learnable parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# Try neural network on a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zeroing gradient buffers of all parameters, because the gradient is accumulative otherwise
net.zero_grad()
# Backpropagate with random gradients
out.backward(torch.randn(1, 10))

# Now for the loss function
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
# Compute mean square error (MSE) loss
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# For illustration, let's fullow a step backwards
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backpropagation
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weights
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

print("Tutorial section 3, Neural networks, complete")
'''

#-----------------------------------------------------------------------
# Training a classifier
# An image classifier
print("Tutorial section 4, Training a classifier, start")


# 1. Load and normalize CIFAR10
print("1. Load and normalize CIFAR10 - start")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Show some training images for fun
# Note: For some reason, this doesn't seem to work
'''
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
print("Show images for fun")
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
'''

print("1. Load and normalize CIFAR10 - Complete")

# 2. Define a Convolutional Neural Network
# Note, this is a copy of the neural network from before
# Define a neural network
print("2. Define a Convolutional Neural Network - Start")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

print("2. Define a Convolutional Neural Network - complete")

# 3. Define a Loss function and optimizer
print("3. Define a Loss function and optimizer - Start")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("3. Define a Loss function and optimizer - Complete")

# 4. Train the network
print("4. Train the network - Start")
for epoch in range(2):  # loop over the dataset multiple times
  print("Epoch: " + str(epoch))
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
      running_loss = 0.0

print('Finished Training')

# Save trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

print("4. Train the network - Complete")

# 5. Test the network on the test data
print("5. Test the network on the test data - Start")
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Load back saved model
# note: saving and re-loading the model not necessary here, done for illustration
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# Let's see what the NN thinks the images are:
outputs = net(images)

# The outputs are energies for classes, the highest energy is which class the NN thinks the image belongs to
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

print("5. Test the network on the test data - Complete")


print("Tutorial section 4, Training a classifier, complete")


#----------------------------------------------------------------------
# Print runtime
#end_t = time.time()
#runtime = end_t-start_t
print("\nRun finished\n--------------------------------------------")
#print("Runtime: " + str(runtime) + " seconds")