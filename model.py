# Adapted partially from https://github.com/aymericdamien/TensorFlow-Examples

import tensorflow as tf
import numpy as np
import random

LEARNING_RATE = 0.01
EPOCH_COUNT = 10000

TRUE_WEIGHTS = np.array([-1, 0.5, 0.33, 0.5, 0.75])
TRUE_BIASES = np.array([0.5, -0.7, 1, .75, -0.5])

# Linear function to model
def f(x):
	return np.multiply(x, TRUE_WEIGHTS) + TRUE_WEIGHTS

# Generate some dummy data
X_train = np.random.rand(10, 5)
Y_train = np.apply_along_axis(f, 1, X_train)

# TF graph input
X = tf.placeholder("float64")
Y = tf.placeholder("float64")

# TF weights
W = tf.Variable(np.random.rand(5), name = "weight")
b = tf.Variable(np.random.rand(5), name = "bias")

prediction = tf.add(tf.multiply(W, X), b)

cost = tf.reduce_sum(tf.pow(prediction - Y, 2)/(2*X_train.shape[0]))
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(EPOCH_COUNT):
		for (x, y) in zip(X_train, Y_train):
			sess.run(optimizer, feed_dict={X: x, Y: y})

		if (epoch % 500) == 0:
			cost_computed = sess.run(cost, feed_dict={X: x, Y: y})
			print("Epoch: %s, Cost: %s " % (epoch, cost_computed))

	W_computed = sess.run(W, feed_dict = {X: x, Y: y})
	b_computed = sess.run(b, feed_dict = {X: x, Y: y})
	print("Results: W=%s, b=%s " % (W_computed, b_computed))



