'''
Perceptron Learning for Classification
Author - Shreyas Kolpe
Date - 3/7/2018
'''

import numpy as np
import math

# Calculate the sigmoid applied element-wise to a vector/matrix
def sigmoid(vec):
	return (np.exp(vec)/(1+np.exp(vec)))

# Calculate the gradient with the current weights
def gradient(X_features, y_labels, weights, num_samples):
	grad = (-y_labels*sigmoid(-y_labels*X_features.dot(weights))).transpose().dot(X_features)
	grad/=num_samples
	return grad

# Read data
file = open('classification.txt','r')
X_features = []
y_labels = []
for line in file:
	values = line.split(',')
	X_features.append([float(values[0]), float(values[1]), float(values[2])])
	y_labels.append(float(values[4]))	
file.close()

# Load training data and labels
X_features = np.array(X_features)
y_labels = np.array(y_labels)
num_samples = X_features.shape[0]
# Padding X with x0 as a column of 1s
X_features = np.hstack((np.ones((num_samples,1)),X_features))

# Tunable parameter learing rate
learning_rate = 0.01

# Initialize weights at guess
weights = np.array([0.,0.,0.,0.])

# Run for 7000 iterations
for i in range(7000):
	# Compute the gradient
	grad = gradient(X_features, y_labels, weights, num_samples)
	# Update the weights
	weights-=learning_rate*grad

# Estimate the hypothesis with learned weights 
hypothesis = sigmoid(X_features.dot(weights))
# Predict label based on threshold of 0.5
predicted_labels = np.where(hypothesis >= 0.5, 1,-1)
# Estimate accuracy
accuracy = np.count_nonzero(predicted_labels == y_labels)/num_samples

print("Learned weights")
print(weights)
print("Accuracy - ", accuracy)