---
layout: post
title: "ML1: Logistic Regression"
description: "Machine learning basics (Logistic Regression)"
modified: 2019-06-01T15:27:45-04:00
tags: [machinelearning, statistics, math]
image:
  path:
  feature:
  credit:
  creditlink:
---


# Inside a Logistic Regression

Lets start by breaking down logistic regression. The name says it all, lets go through both parts first "what is regression" then expand on the topic of "what is logistic regression".

## Regression

**Regression** is a measure of the relationship between a dependent (eg. output) and independent variable (e.g. inputs, weight, age, etc). A good way to completely explain regression is by going through the most basic method in statistical learning which is a **linear regression**.

> In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.


<figure>
	<a href="https://qph.fs.quoracdn.net/main-qimg-3b0d7655ac76edf1241f97015ee755b4"><img src="https://qph.fs.quoracdn.net/main-qimg-3b0d7655ac76edf1241f97015ee755b4" alt=""></a>
	<figcaption><a>Linear regression representation</a>.</figcaption>
</figure>

Lets go over this figure

### Hypothesis of a linear model

Equation and hypothesis for linear model is as follows. Y = a + bx

* Y is our output or dependent variable. (e.g. housing price)
* X is our input or independent variable(e.g. number of windows)
* **a** and **a** are the coefficients that is needed to be changed in order to make predictions. **a** is our intercept and **b** is the slope of the regression.

Now that we have established the hypothesis for a linear equation and lets see how we find the line that best represents the data.

### Cost Function (Linear regression)

For us to know how to fit a line, we need to establish a cost. Cost can be likened to something like a goal or an objective we would like to meet. Likewise in this case we want to minimize the cost or in other terms cost. 

Linear regression uses somehting called OLS (ordinary least squares). Going back to the graph above $y-\hat{y}$ is simply the difference between the actual value and our line. We then square this $(y-\hat{y})^2$ to get rid of negative values. Then lastly we sum all the instances which results in the the formula below

$$
\sum_{n=1}^n (\hat{y} - y)^2\,.
$$

And the objective is minimizing this

## Logistic Regression

Now that we have established regression, lets go back to **what is logistic regression?** Logistic regression is a classification model, where the output is either 1 or 0. Linear regression as discused gives a unbounded value (infinity) but for logistic we need the outcome to be a probability score that lies between 0< P(x)<1. 

Logistic comes from `logit`, this is the function that gives us the probability of weather an event happens or not. The logit function transforms our continious output and bounds it between 0 and 1.

logit = log(p/1-p)

 Funny thing is we actually need the inverse of the logit function which is called the sigmoid function. (Too lazy to write this down in MathJax so I am just posting a picture of what I just wrote on paper)

<figure>
	<a href="/images/ML1_logit.jpg"><img src="/images/ML1_logit.jpg" alt=""></a>
	<figcaption>Two images.</figcaption>
</figure>

So to cap it off, the main difference between a linear regression and logistic regression is the presence of a activation function and this is the sigmoid function {% raw %} $$ \sigma(z) =  \frac{1}{1 + e^{-(z)}}$$ {% endraw %}  

{% raw %} This function is the basis of logistic regression, where our linear equation will be fed into the sigmoid function $$ \sigma(w^T x + b) =  \frac{1}{1 + e^{-(w^T x + b)}}$$ {% endraw %}. From here we will receive an output between 0 and 1, it is common practice to apply a decision boundary at this probability output to limit the value to either 0 or 1. Default value when dealing with dichotomous variable is setting the decision boundary at 0.5, so if y>0.5 then output is 1.

### Cost Function (Logistic regression)

Same case with linear regression, we need to establish a Cost function that we can minimize. The cost function for logistic regression can't be the same function used in linear regression. We need a function that is convex in order for the optimization process to converge on a global minimum. Logistic regression does not uses OLS (Ordinary Least Square) for parameter estimation, instead it uses maximum likelihood estimation (MLE).

Lost function is as follows: {% raw %} $$ \mathcal{L}(\hat{y},y) = -(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}))$$ {% endraw %}

So what we need is a function to measure the amount of cost whenever we get a prediction right or wrong (1 or 0).

Two instances to work with here:
* if Y = 1 We want to have large cost if we predict close to 0 and lower cost if we predict close to 1.
* if Y = 0 Same case, we want a large cost if we predict close to 1 and lower cost if we predict close to 0.

if Y = 1, we need to use a function that will map cost accordingly and that is $-log(a^{(i)})$

<figure>
	<a href="/images/ML1_log1.PNG"><img src="/images/ML1_log1.PNG" alt=""></a>
	<figcaption>yes, this is a screenshot from a google search result of the function</figcaption>
</figure>

And if Y = 0, the appropriate function is $-log(1-a^{(i)})$

<figure>
	<a href="/images/ML1_log1.PNG"><img src="/images/ML1_log2.PNG" alt=""></a>
</figure>

okay now we have the 2 main parts that create out cost function. We just need a mathetematical way to control which function to use depending on weather the Y is 1 or 0. 

* Y = 1: $y (-log(a^{(i)}))$
* Y = 0: $(1-y) (-log(1-a^{(i)}))$

So when y is 1 it cancels out $-log(1-a^{(i)})$ and when 0 it removes $-log(a^{(i)})$. So if we sum both of this together we get:

$y^{(i)}(-log(a^{(i)})) + (1-y^{(i)}) (-log(1-a^{(i)}))$

Finally this gives us our cost function 

$\mathcal{L}(\hat{y},y) = -(y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})) $

We have our linear equation that is fed into a function squeezes the output between 1 and 0,then we adjust our weights or coefficients by minimizing our cost. I'll write about optimization through gradient descent in another post (just gotta brush up on calculus for that one).
