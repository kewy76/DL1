# Kate Williams
# 7/3/2018

# Assignment: calculate the function (a^2+b)*c where a, b, and c are matrices
# Provide the appropriate comments for each operation in the program

import tensorflow as tf  # Import tensorflow
import random

tf.InteractiveSession()  # Open a tensorflow session

a = tf.fill([2, 2], random.randint(1, 5))  # Create the matrices a, b, and c
b = tf.fill([2, 2], random.randint(1, 5))
c = tf.fill([2, 2], random.randint(1, 5))

print("a = ", a.eval())  # Print a, b, and c
print("b = ", b.eval())
print("c = ", c.eval())

# Without matrix multiplication
print("a^2 = ", tf.multiply(a, a).eval())  # a * a == a^2

print("a^2 + b = ", tf.add(tf.multiply(a, a), b).eval())  # a^2 + b

print("(a^2 + b) * c = ", tf.multiply(tf.add(tf.multiply(a, a), b), c).eval())  # (a^2 + b) * c

# With matrix multiplication
print("a^2 = ", tf.matmul(a, a).eval())  # a * a == a^2

print("a^2 + b = ", tf.add(tf.matmul(a, a), b).eval())  # a^2 + b

print("(a^2 + b) * c = ", tf.matmul(tf.add(tf.matmul(a, a), b), c).eval())  # (a^2 + b) * c

