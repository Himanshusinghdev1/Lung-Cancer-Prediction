import tensorflow as tf
import numpy as np

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for Metal support
print("\nPhysical devices:")
print(tf.config.list_physical_devices())

# Simple matrix multiplication test
print("\nTesting basic operations:")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.matmul(a, b)
print("Matrix multiplication result:\n", c.numpy())

# Basic neural network operation
print("\nTesting neural network operations:")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Successfully loaded MNIST dataset")
print("Training data shape:", x_train.shape)

