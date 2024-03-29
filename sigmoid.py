import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-np.array(x)))
def relu(x):
	return np.maximum(0, np.array(x))
def leaky_relu(x , alpha=0.01):
	return np.maximum(alpha* np.array(x), np.array(x))
def tanh(x):
	return np.tanh(np.array(x))
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
print("sigmoid:",sigmoid(random_values))
print("ReLU:", relu(random_values))
print("Leaky ReLU:", leaky_relu(random_values))
print("Tanh:",tanh(random_values))