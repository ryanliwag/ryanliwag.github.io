---
layout: post
title: "Building an intuition towards GAN (Tensorflow)"
date: 2018-02-14 9:32:20 +0300
description: Building a highlevel overview of GAN and different types of GAN's
---

Post to help myself better articulate and understand GAN's and thier current state.

## [Generative Adversial Networks](https://arxiv.org/pdf/1710.07035.pdf)


GANs are a technique used for both semi-supervised and unsupervised learning. They are trained through a pair of networks in competition with each other. Gan's analogy: there are 2 networks, one is the counterfeiter and the other one is the cop. The cop is trained to identify real from fake, while the counterfeiter tries to build more convincing fakes. The cop is known as the Discriminator and the counterfeiter is the Generator. In GANs the Generators have no access to real images, it only learns through the discriminator.

### Types of GAN's

There are more variations of GAN's, these are just the few that I read

#### Fully Connected GANs

![GAN]({{site.baseurl}}/assets/img/project-understanding-GAN/gan.png)

The first types of GAN architectures used fully connected neural netwokrks for both the generator and discriminator.

#### Convolutional GANs and Deep Convolutional GAN's

Similar to Fully connected GAN's but convolutional layers are added. Convolutional GAN's are well suited for image data.

#### Conditional GANs

![CGAN]({{site.baseurl}}/assets/img/project-understanding-GAN/cgan.png)

These Types of GAN's implement a conditional setting where both generator and discriminator are conditioned to some class label y. Its architecture is to feed y inyo both Generator and discriminator additional input, such that y and X will combine in a hidden representation.

#### GANs with Inference Models

![BIGAN]({{site.baseurl}}/assets/img/project-understanding-GAN/BiGan.png)

Previous implementations of GANs did not have a way to map the input obeservation to vector. Here the generators consist of 2 netowkrs, and encoder and decdoer. The encoder is what wuold be referred to as the inference network. They are jointly trained, and given to the discriminator to receive pairs of (x,z) vectors.
