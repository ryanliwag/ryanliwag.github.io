---
layout: post
title: "Building an intuition towards GAN (Tensorflow)"
date: 2018-02-14 9:32:20 +0300
description: Building a highlevel overview of GAN and different types of GAN's
---

## [Generative Adversial Networks](https://arxiv.org/pdf/1710.07035.pdf)


GANs are a technique used for both semi-supervised and unsupervised learning. They are trained through a pair of networks in competition with each other. Gan's analogy: there are 2 networks, one is the counterfeiter and the other one is the cop. The cop is trained to identify real from fake, while the counterfeiter tries to build more convincing fakes. The cop is known as the Discriminator and the counterfeiter is the Generator. In GANs the Generators have no access to real images, it only learns through the discriminator.

### Types of GAN's

#### Fully Connected GANs

The first types of GAN architectures used fully connected neural netwokrks for both the generator and discriminator.

#### Convolutional GANs and Deep Convolutional GAN's

Similar to Fully connected GAN's but convolutional layers are added. Convolutional GAN's are well suited for image data.

#### Conditional GANs

These Types of GAN's implement a conditional setting where both generator and discriminator are conditioned to some class label y. Its architecture is to feed y inyo both Generator and discriminator additional input, such that y and X will combine in a hidden representation.

#### GANs with Inference Models


#### Adversial Autoencoders
