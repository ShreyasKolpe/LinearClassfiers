# Linear Classifiers

Linear Classifiers try to draw a hyperplane in the dimensions of the data in order to classify. For 2D data, this is a line. The word 'linear' here refers to the fact that in the hypothesis for the decision boundary, features only combine linearly. For data that is not linearly separable, the hypothesis will need to include other polynomial terms.

The Perceptron, also known as Linear Classifier and Logistic Regression are used to classify data points into two or more classes. Both try to draw a hyperplane to separate the data into distinct regions. The Perceptron is satisfied with finding some boundary that classifies the data correctly while the Logistic Regressor optimizes a boundary via gradient descent to minimize the classification loss. 

## Perceptron

As mentioned, it finds a linear boundary to separate the data if it is linearly separable. If the data is not, it will never be able to find a line, and hence will not stop. We might be satisfied with just finding a boundary that gives the fewest misclassifications. This is called the Pocket algorithm presumably because we run several iterations and keep the best solution in our 'pocket'.

In Perceptron, we start with assigning some initial values to the parameters of the hyperplane w'x = 0. If a point x' evaluates to w'x' > 0, it is classified as the positive class label, while if w'x' < 0, it is assigned the negative class label. 
The algorithm runs the following loop, 
1. Plugs in every data point x to compute w'x.
2. If the class labels are correct, stop
3. Else, for every point that is misclassified, update the parameters w by moving a small step towards the correct region.

The program `perceptron.py` is organized as follows –
• Method to calculate the number of violated constraints (misclassified points) and the indices of the data points which violate the constraint – `violated_constraints(X_features, y_labels, weights)` – returns `num_violated`, `violated_indices`.
Estimate hypothesis as `w’x`. If points are misclassified the sign of `hypothesis[i]` and `y_labels[i]` are opposite. `dissonance` is the element- wise product of `hypothesis` and `y_labels`. `violated_indices` is a logical array of all indices where `dissonance[i]` is negative. `num_violated` counts the occurrences of `True` in `violated_indices`.
• The code next reads the data and loads the labels into `numpy` `ndarrays` `y_labels` and the training data into `X_features`. `X_features` is padded with x0 as a column of 1s.
The tunable parameters `learning_rate` and initial value of weights is assigned.
The stopping condition for the iterative algorithm is based on the number of violated constraints being less than the threshold. In this case, it is set to 1000th of the number of data points.
The algorithm loops until the number of misclassifications is less than `threshold`, by calling `violated_constraints()`. At the same time, it loops over all the data points and updates weights according to whether the point is misclassified or not. If so, the magnitude of the update is determined by the product of `learning_rate` and `X_features[i]`, while the sign of the update is determined by the true labels `y_labels[i]`.

### Pocket

The implementation `pocket.py` mostly follows the perceptron.py program with the exception that the labels make the data not linearly separable. The algorithm iterates 7000 times, appending `num_violated` to `violated_counts`. `best_weights` keeps track of the weights that correspond to the iteration that gives the least value for violated constraints – `min_violated`. The number of misclassifications is plotted against the number of iterations, in steps of 100.

![]()

## Logistic Regression

Logistic Regression is so called, despite the fact that it is used for classification is that it gives a continuous value from a probability distribution, when using the sigmoid function. This is interpreted as the probability that the data point belongs to the positive class. The algorithm finds a 'better' boundary than the Perceptron, in the sense that it is not satisfied by getting class labels correct for the training set. It tries to find a boundary that minimizes a cost function - cross entropy loss.

To get the best parameters/weights, we start with an initial set and iteratively move towards better solution by gradient descent.

The code `logisticRegression.py` is organized as follows –
• Method to calculate the sigmoid function –
`sigmoid(vec)` – takes a `numpy` `ndarray` vec and returns a matrix of the same shape with the sigmoid function applied element-wise.
• Method to calculate the gradient given the data, labels and weights – `gradient(X_features, y_labels, weights, num_samples)` – Returns a `numpy` `ndarray` of dimensionality equal to the number of features in `X_features`. Gradient is calculated in an efficient vectorized manner
• The code loads the data into `numpy` `ndarrays` `X_features` and `y_labels`
with `X_features` padded with the feature x0 as a column of 1s.

The `learning_rate` is initialized (here set to `0.01`) and the weights are initialized arbitrarily as all 0s.
The algorithm loops for `7000` iterations, each time calculating the gradient `grad` and updating the weights according to the gradient descent formula.
The learned weights are used to estimate the probability of the data points belonging to the positive class in `hypothesis`. The predictions are collected in `predicted_labels` based on thresholding `hypothesis` at `0.5`.
Finally, the accuracy is reported as the 0/1 misclassification error.


