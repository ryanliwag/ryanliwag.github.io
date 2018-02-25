---
layout: post
title: "Logistic Regression"
date: 2018-02-23 10:32:20 +0300
description: Just some machine learning basics
---

It has always bothered me to use scikit-learn library to some extent, not because it's bad but because I can't seem to understand the underlying principle of the theories behind each machine learning procedure. This post, I will try to concisely explain how logistic regression operates at least in a mathematical level which is low enough so that if ever I need to, I can reproduce the code in numpy.

## Logistic regression

### Overview

Logistic regression is comparable to how a neural network operates, in fact, it could even be identified as a 1 layer neural network. Logistic regression can be used for either binary or multiclass classification (one for all method or softmax). So despite being likened to a neural network, it is still a linear model, due to its calculation always depending upon the sum of the inputs and parameters.

### Logistic Regression and Linear Regression

To start off, the main difference between a linear regression and logistic regression, is the implementation of the sigmoid function at its linear formula which is {% raw %} Y = wX + b {% endraw %}, this same formula is for calculating the slope of a line and this is what linear regression does, which is fitting a line that minimizes error through the data. But this gives a continuous output, on the other hand, logistic regression gives a binary output. This binary output is caused by a sigmoid function applied to the linear formula that gives the probabilities of the prediction. The sigmoid function which is {% raw %} $$ \sigma(z) =  \frac{1}{1 + e^{-(z)}}$$ {% endraw %}  has the characteristic of an s-shaped curve.

{% raw %} Sigmoid will be applied as such $$ \sigma(w^T x + b) =  \frac{1}{1 + e^{-(w^T x + b)}}$$ {% endraw %}. This function is the basis of logistic regression, the output of this function is used to calculate the cost function for optimizing and once wiegths are optimized this is the model used for predicting.

Python Code
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Forward propagation process
A = sigmoid(np.dot(w.T, X) + b)
```

### Cost Function and Optimization

Its the main objective of any machine learning model is to optimize and this is can be done through minimizing the loss function through a process of gradient descent. The cost function can in simpler terms be stated as the summary of the loss function.

Loss Function: {% raw %} $$ \mathcal{L}(\hat{y},y) = -(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$$ {% endraw %}

Cost Function:  {% raw %} $$ \mathcal{J}(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y},y)$$ {% endraw %}

The process of decreasing the cost function is most commonly known as gradient descent. There is a bit of calculus involved with getting the gradients but that's about it.

The gradient of the loss with respect to w: $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$

The gradient of the loss with respect to b: $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$

When these grads are calculated they are then multiplied by the set learning rate and is subtracted from the previous weight values. This update process repeats for as many iterations that I set.

```python
A = sigmoid(np.dot(w.T, X) + b) # compute activation
cost = (-1/m) * (np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A))))   # compute cost

# BACKWARD PROPAGATION
grad_w = (1/m) * np.dot(X, (np.subtract(A,Y)).T)
grad_b = (1/m) * np.sum(A - Y)

# Updating the w and b values
w = w - (learning_rate * grad_w)
b = b - (learning_rate * gra_b)
```

so the steps required to complete the process is

1- Initialize the weights for (w & b), they can be random values or could be intialized at zero.

2- Calculate the gradient, change in the cost function when their values are changes by a small amount from the original. This improves the values of (w & b) in the direction in which the cost function is minimized.

3- Adjust the weights using the gradients to reach optimial value.

4- Use the weights to predict and get a new cost.

5- Steps 2-3 will repeat for number of set iterations, and will keep adjusting the weigths, ideally untill no error can no longer be significantly reduced.

Full Python code for [logistic regression binary classification] implemented on numpy on the wine dataset
