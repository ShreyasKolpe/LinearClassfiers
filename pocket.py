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
	y_labels.append(float(values[4]))
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

# Keep track of counts of violated constraints
violated_counts = []

# Keep track of the minimum violations and best weights
min_violated = num_samples
best_weights = weights

for i in range(7000):
	num_violated, violated_indices = violated_constraints(X_features, y_labels, weights)
	violated_counts.append(num_violated)
	# Check if weights are better
	if num_violated < min_violated:
		min_violated = num_violated
		best_weights = weights
	if(violated_indices[i%num_samples] == True):
		weights+=learning_rate*y_labels[i%num_samples]*X_features[i%num_samples]

print('Best weights at stopping')
print(weights)
print("Accuracy - ",(num_samples-min_violated)/num_samples)

# Plot the misclassified points
x = list(range(0,len(violated_counts),100))
y = [violated_counts[i] for i in x]

fig = plt.figure("Constraints Violated")
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Iterations')
ax.set_ylabel('Number of misclassified points')
ax.plot(x,y,'r')
plt.show()