import numpy as np 
import pandas as pd
import tensorflow as tf
import sys

TF_ENABLE_ONEDNN_OPTS=0

"""
Single Layer Perceptron w/ MNIST Data

Data Representation;
-Each hand written digit is a 28 x 28 grid of pixel in grayscale

-The 28 x 28 grid is flattened to a 1-D array of 784 values
-Then the values are converted from grayscale values (0-255), with 0 being black and 255 being completely white
to a (0-1) to keep values simpler for the perceptron
-The set of data is also split into x_train and x_test, for training the perceptron then testing
	The final shape of X-train is (784, 59000)

-Each digit is also given an 'answer' or what the digit actually is, used to check the perceptrons answer
-This data is one-hot encoded or transformed into a length ten array, with each digit representing the number (0-9)
-So if the digit is 3, the array becomes [0,0,0,1,0,0,0,0,0,0]
	The final shape of Y_train pre-one-hot is (59000,) and post one-hot encoding (done in back_prop) is (10,59000)

PERCEPTRON LAYERS
The perceptron below is single layered, meaning it has an input layer ([0]), a single hidden layer ([1]) and a output layer ([2])

LAYER 0 - Input Layer:
This layer is simply one image/digit. So in this case there 784 nodes in the input later, each representing one pixel in the digit.
As mentioned above each pixel is on a scale of (0-1)

LAYER 1 - Hidden Layer:
This layer has 10 nodes, each of which are individually connected to each node in the input layer.
The value of these nodes range from 0-1 and calulated using the Weights and Biases and previous layer nodes.

LAYER 2 - Output Layer:
This layer also has 10 nodes, each one representing a digit 0 - 9
The value of each node represents the models confidence the image is that number, so:
an output of [0.1, 0.0, 0.3, 0.1, 0.0, 0.9, 0.2, 0.0, 0.1, 0.2] would mean the model is quite confident the image is a 5
an output of an array with multiple high numbers or none, means the model is not confident
This model is also calculated similarly with weights and biases


Forward Propogation:
The weights and biases mentioned above are represented in W1, W2, and b1, b2
Each weight represents a single connection from one node onother across layers
The shape of W1 is (10, 784) since there are 10 nodes in layer 1 and 784 in layer 0
b1 is only (10,1) since each node in layer one has one bias.
So, first The value of every Input node is multplied by its weight then the nodes bias is added.
These values are then put in ReLU to make them better for the model. This all repeates again for the first layer to the output layer.
However instead of ReLu, softmax is used. Soft max limits the range from (0,1).
This answer is stored as A2, and can be compared to Y, in back propogation


Back Propogation:
**Note back propogations requires the following inputs:
Z1, A1: Activations and pre-activation (Z1) from the first layer (hidden layer).
Z2, A2: Activations and pre-activation (Z2) from the second layer (output layer).
W1, W2: Weights of the first and second layers, respectively.
X: Input data.
Y: True labels (ground truth).
m: Number of examples in the batch (batch size).**


Now that the data has been passed through the layers, we have a guess for what the image is
This is compared to the corresponding answer in Y. lets say we have the following guess & answer:
  A = [0.1, 0.0, 0.3, 0.1, 0.0, 0.9, 0.2, 0.0, 0.1, 0.2]
  Y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

As you can see the bot correctly guesses the image as being a 5, but it wasn't perfect so we still want to update
our parameters, which are the weights and biases. First the difference is found as dZ2
dZ2 = [0.1, 0.0, 0.3, 0.1, 0.0,-0.1, 0.2, 0.0, 0.1, 0.2]

With this, dW2 is calculated which is the gradient of the loss (or difference) with respect to the weights of the second layer (W2)
db2 is also similarly calculated and is the gradient of the loss with respect to the biases of the second layer (b2)
We continue on to calculate dZ1 which is the gradient of the loss with respect to the pre-activation value of the hidden layer (Z1)

Using dZ1 we can similarly find dW1 and db1

Then dW1, db1, dW2, and db2 can be used to update the parameters.



Updating Parameters:

The above four gradients are then used to update the parameters.
This is relatively simple and take the initial four parameter, the gradients and alpha (the learning rate)
To update, say W1, we would do:
W1 = W1 - alpha * dW1

Finally once all parameters have been updated, the perceptron restarts with the next handwritten digit.

*All specfic math can be found in the respective function*
"""



"""
Parameter Initialization:

W1 & W2: The wieghts for the connection between all 0th and 1st layer nodes, and all 1st and 2nd layer nodes respectively

b1 & b2: The biases for the first and second layer nodes respectively
"""
def init_params():

    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784)) #Shape: (10,784)
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10) #Shape: (10, 1)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20) #Shape: (10, 10)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784)) #Shape: (10, 1)

    return W1, b1, W2, b2


"""
ReLU:
x: 
x >= 0;  =x
x < 0; = 0
"""
def ReLU(Z):
	#print('relu')
	return np.maximum(Z, 0)


"""
Softmax:
	Z is array of numbers

	return e^Z[i]/sum of (e^Z[0:n])
"""
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


"""
Forward Propogation:
Z1 are the pre-activation values of the hidden layer, calculated by taking all values of x and multiplying by their respective weight and adding the correct bias.
Thanks to numpy and matrix math this can be done with a simple dot product and addition. Keep in mind it is doing it for all 10 nodes in the hidden layer.

Z1 is plugged into ReLU and then the same process is repeated to get the pre-activation values (Z2) for the output layer.
Instead of ReLU, Softmax is used to normalize the values to a range of 0-1.

All of Z1, A1, Z2, and A2 are returned so back propogation can use them.
"""
def forward_prop(W1, b1, W2, b2, x):
	#print('forward propped')
	#print("IN forward_prop")
	#print("x shape (X_train)", x.shape)
	#print("W1 shape:", W1.shape)
	Z1 = W1.dot(x) + b1  
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = softmax(Z2)
	
	return Z1, A1, Z2, A2


"""
ReLU Derivative:
Simple way to calculate the ReLU derivative since if Z is less than 0, it stays 0 w/ a slope of 0 and
if Z > 0 then it is a linear slope so derivative is just 1.
Conveniently, True and False also represent 1 and 0 so all that is needed is Z > 0
"""
def deriv_ReLU(Z):
	#print('derive relu')
	return Z > 0

"""
One-Hot Encode

This takes a number and One Hot encodes it.
The first line creates an array of all zeros of length 10
So if Y has 1000 elements, one_hot_Y is [[10 0s],[10 0s], + 998 more]

The second line updates the array so that the index that corresponds to the value in Y is set to 1
np.arrange(Y.size) creates an array of indices for the rows, then the proper index in one_hot_Y is set to 1
the array is transposed so the size is (10, 59000) instead of (59000, 10). This is purely so future calculations are easier
"""
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y



"""
Backwards Propogation
Y is first one-hot encoded

then dZ2 is is simply the difference between the guess (A2) and answer (one_hot_Y)

dW2 = 1 / m * dZ2.dot(A1.T) - calculates the loss gradient for W2
this is just the dot product of dZ2 and A1.T normalized by 1/m
db2 is calculated by summing dZ2 along the examples axis, collapsing it into a column vector of the same shape as b2, then 1/m averages the bias gradients

dZ1: W2.T.dot(dZ2) gives how much of the error in the second layer (dZ2) back propogates to the first layer, 
the derivative of ReLU is then applied as a mask so negative values are set to 0 and postive values remain unchanged

dW1 is calculated the same way as dW2 but with dZ1 and X instead of dZ2 and A1

db1 is also calculated the same as db2 but with dZ1 instead of dZ2

*Note all d's are the same shape as their repective gradient, so dW1 is the same shape as W1 and dZ1 is the same shape as Z1 etc.
"""
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


"""
Update Parameters:
The math here is very simple, we take the losses and subtract them from the weights and biases, but just doing that would change the perceptron too fast,
so we multiply the change by a small alpha value, of learning rate. This way the perceptron is slowly changed each iteration and can better reach the local minimum solution
"""
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	W1 = W1 - alpha * dW1
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2
	return W1, b1, W2, b2

def get_predictions(A2):
	return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
	return np.sum(predictions == Y) / Y.size


"""
Gradient Descent
The core function. Calls forward prop then backprop and then updates the parameters for each iteration.
Every 50 iterations the new accuracy is printed so the improvements can be see over time
"""
def gradient_descent(X, Y, alpha, iterations, m):
	W1, b1, W2, b2 = init_params()
	for i in range(iterations):
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2,W1, W2, X, Y, m)
		W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		if i % 10 == 0:
			print("Iteration: ", i)
			accuracy = get_accuracy(get_predictions(A2), Y)
			print(f"Accuracy: {accuracy:.4f}")
			sys.stdout.flush()
	return W1, b1, W2, b2


def main():
	sys.stdout.reconfigure(encoding='utf-8')

	#getting data from tensorflow
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


	x_train = x_train.reshape(-1, 28*28) 
	x_test = x_test.reshape(-1, 28*28)

	
	data = np.concatenate((y_train.reshape(-1, 1), x_train), axis=1) 

	#shuffle data so its a random order
	np.random.shuffle(data)

	#saving some data for testing
	data_dev = data[0:1000].T  
	Y_dev = data_dev[0]         
	X_dev = data_dev[1:]        
	X_dev = X_dev / 255.0       

	#setting up rest of data for training
	data_train = data[1000:].T  
	Y_train = data_train[0]     
	X_train = data_train[1:]    
	X_train = X_train / 255.0   

	# Get the number of training samples
	_, m_train = X_train.shape

	print("Shape of X and Y train:")
	print("X_train:", X_train.shape)
	print("Y_train:", Y_train.shape)
	sys.stdout.flush()


	#The final values of W1, b1, W2, and b2 can be used for testing
	W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 1000, m_train)



if __name__ == "__main__":
	main()














