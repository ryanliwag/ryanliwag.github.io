---
layout: post
title: Gist of Convolutional Neural Networks
date: 2018-01-25 19:32:20 +1200
description:  A High Level understanding of Convolutional Neural Networks
img:  project-understanding-CNN/CNN-example.png
tags: [Neural Networks, CNN, self-learning]
---

### Overview

Convolutional neural networks is different algorithms working together.The main differecen compared to a normal neural network is the preprocessing done at the start of the model. CNN's are divided into 2 steps.

1. Feature engineering/extractions
2. Train the model to map images and extracted features to labels


### 1. [Convolution](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)

The convolution process in applying an image filter on top of the input image. This filter(kernel) is dragged(convolving) across the image pixel by pixel, this results in producing a filtered subset out of the input image.

Common parameters during convolution layers. 
- Kernel size (3x3, 5x5 Size of the matrix you want to convolve over the image)
- Kernel Type (filtering types such as edge detection, sharpen, etc)
- stride (number of pixel increments the kernel takes when moving accross)
- Padding (add a layer of 0s to the outside of the image, so that the kernel properly passes the edge.)
- Output Layers (How many different kernels are applied to the image?)

The output here is called a convolved feature or feature map.

### 2. Activation Function (RELU)

Applying activation function are commonly applied in neural networks. Without them only linear transformations would be used and models would be unable to learn complex mapping funcitons. Sigmoid and Tangent have a limitation in deep neural netoworks which is due to the [vanishing gradient problem](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/). Another problem is that these functions saturate when having large or small values squeezed to 1 and 0. Limited sensitivity becomes a challenge for the algorithm to learn.

Enter [Relu](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (Rectified linear activation unit). Relu solves the saturation problem and prevents the gradients from vanishing. Relu also intorduces computational simplicity with the functions output simply being output = Max(0, input).

### 3. Max Pooling

Max pooling being very self explanatory, this is where we take the max of a section of an image then pool them into the highest value (Ex: MaxPool(5,6,2,9) = 9). Max pooling has few of the same parameters as the Convolution procedure. Other types of pooling are Sum and average. Purpose here is to further reduce the feature set size.

### 4. Fully Connected Layers

These Fully connected layers are the ones responsible for the classification, where as feature extraction and processing is handled by the previous steps. This part here is similar to any other normal neural network.
