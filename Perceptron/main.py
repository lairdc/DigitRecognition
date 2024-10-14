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
	The final shape of Y_train pre-one-hot is (59000,) and post one-hot (done in back_prop) is (10,59000)







"""



"""
Parameter Initialization:
"""
def init_params():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))


    print("Param shapes:")
    print("W1: ", W1.shape)
    print("b1: ", b1.shape)
    print("W2: ", W2.shape)
    print("b2: ", b2.shape)
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
softmax:
	Z is array of numbers

	return e^Z[i]/sum of (e^Z[0:n])
"""
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


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



def deriv_ReLU(Z):
	#print('derive relu')
	return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_Y = one_hot(Y)
    print("Y shape:", one_hot_Y.shape)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2



def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
	#print('params updated')
	W1 = W1 - alpha * dW1
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2
	return W1, b1, W2, b2

def get_predictions(A2):
	#print('getting predictions')
	return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
	#print('getting accuracy')
	#print(predictions, Y)
	return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, m):
	#print("In gd x:", X.shape)
	#print('gradient descending')
	W1, b1, W2, b2 = init_params()
	for i in range(iterations):
		Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2,W1, W2, X, Y, m)
		W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
		if i % 50 == 0:
			print("Iteration: ", i)
			accuracy = get_accuracy(get_predictions(A2), Y)
			print(f"Accuracy: {accuracy:.4f}")
	return W1, b1, W2, b2


def main():
	# Reconfigure stdout to avoid encoding errors on Windows
	#print("descending")
	# Reconfigure stdout to handle utf-8 encoding
	sys.stdout.reconfigure(encoding='utf-8')

	# Load the MNIST dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	# Reshape the MNIST data for compatibility with the provided format
	x_train = x_train.reshape(-1, 28*28)  # Flatten the images from 28x28 to 784
	x_test = x_test.reshape(-1, 28*28)

	# Concatenate train images and labels into a single dataset
	data = np.concatenate((y_train.reshape(-1, 1), x_train), axis=1)  # labels first, then features

	# Shuffle the dataset before splitting
	np.random.shuffle(data)

	# Split the dataset into dev and training sets
	data_dev = data[0:1000].T  # First 1000 examples for dev set
	Y_dev = data_dev[0]         # Labels for dev set
	X_dev = data_dev[1:]        # Features for dev set (all except the first row)
	X_dev = X_dev / 255.0       # Normalize pixel values from 0-255 to 0-1

	data_train = data[1000:].T  # Rest of the examples for training set
	Y_train = data_train[0]     # Labels for training set
	X_train = data_train[1:]    # Features for training set (all except the first row)
	X_train = X_train / 255.0   # Normalize pixel values

	# Get the number of training samples
	_, m_train = X_train.shape

	print("Shape of X and Y train:")
	print("X_train:", X_train.shape)
	print("Y_train:", Y_train.shape)


	#print("descending")
	W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 1000, m_train)

	#print(W1, b1, W2, b2)


if __name__ == "__main__":
	main()














