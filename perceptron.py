'''
Perceptron Learning for Classification
Author - Shreyas Kolpe
Date - 3/7/2018
'''

import numpy as np 
import matplotlib.pyplot as plt

# method to calculate the number of constraints violated(misclassified points)
# and indices into X_features where the constraints are violated
def violated_constraints(X_features, y_labels, weights):
	# Calculate w'x
	hypothesis = X_features.dot(weights)
	# Check if w'x[i] and y[i] are opposite in sign and record i
	dissonance = hypothesis*y_labels
	violated_indices = dissonance < 0
	# Count number of violations
	num_violated = np.count_nonzero(violated_indices)
	return num_violated, violated_indices		

# Read data
file = open('classification.txt','r')
X_features = []
y_labels = []
for line in file:
	values = line.split(',')
	X_features.append([float(values[0]), float(values[1]), float(values[2])])
	y_labels.append(float(values[3]))
file.close()

# Load training data and labels
X_features = np.array(X_features)
y_labels = np.array(y_labels)
# Size of data set
num_samples = X_features.shape[0]
# Padding with x0 as a column of 1s
X_features = np.hstack((np.ones((num_samples,1)),X_features))

# Learning rate that can be tuned
learning_rate = 1.0

# Initialize weights by guessing
weights = np.array([1.,0.,0.,0.])

# Iteration count
iteration_count=0
# Threshold of number of violations for stopping
threshold = num_samples/1000

while True:
	# Iterate over all samples
	for j in range(num_samples):
		num_violated, violated_indices = violated_constraints(X_features, y_labels, weights)
		if(num_violated < threshold):
			break
		# If constraint is violated, update weights
		if(violated_indices[j] == True):
			weights+=learning_rate*y_labels[j]*X_features[j]
		iteration_count+=1
	if(num_violated < threshold):
		break


print('Weights at stopping')
print(weights)
print("Accuracy ", (1-num_violated/num_samples))