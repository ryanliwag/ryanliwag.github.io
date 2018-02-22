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

#### Simple GAN python implementation (MNIST)

GAN's dont need labels, and training the generator is all that matters. Discriminator needs to recieve 2 inputs, one from the generator and another from the original image. Random noise is fed as input towards the generator.



```python
#Model Parameters

epochs = 100000
batch_size = 128
learning_rate = 0.0002

# Network Parameters
input_dim = 784 # 28 x 28
generator_hidden = 256
discriminator_hidden = 256
noise_dim = 100 # Noise input

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, generator_hidden])),
    'gen_out': tf.Variable(glorot_init([generator_hidden, input_dim])),
    'disc_hidden1': tf.Variable(glorot_init([input_dim, discriminator_hidden])),
    'disc_out': tf.Variable(glorot_init([discriminator_hidden, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([generator_hidden])),
    'gen_out': tf.Variable(tf.zeros([input_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([discriminator_hidden])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

    # Network Inputs
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, input_dim], name='disc_input')

    # Build Generator Network
    gen_sample = generator(gen_input)

    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)


    # Build Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                biases['disc_hidden1'], biases['disc_out']]

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    dataset = tf.data.Dataset.from_tensor_slices((X_values))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    batched_dataset = dataset.batch(batch_size)


    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()


    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for i in range(1, epochs+1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = sess.run(next_element)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Train
            feed_dict = {disc_input: batch_x, gen_input: z}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))


```
