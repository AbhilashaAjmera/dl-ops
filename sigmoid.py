import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-np.array(x)))
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
print(sigmoid(random_values))