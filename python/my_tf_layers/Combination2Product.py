import tensorflow as tf
from tensorflow import keras
import numpy as np


class Combination2Product(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        """Called the first time call() is called. input_shape has the shape of each batch"""
        num_inputs = input_shape.as_list()[-1]
        if num_inputs == 1:
            raise Comb2InputError()
        self.num_outputs = int(0.5 * num_inputs * (num_inputs - 1))  # num_inputs choose 2
        #w_init = tf.random_uniform_initializer()
        #self.w = tf.Variable(w_init((self.num_outputs,)), trainable=True)
        self.pairs1 = []
        self.pairs2 = []
        node1 = 0
        node2 = 1
        while node1 < num_inputs - 1:
            self.pairs1.append(node1)
            self.pairs2.append(node2)
            node2 += 1
            if node2 == num_inputs:
                node1 += 1
                node2 = node1 + 1

    def call(self, inputs, **kwargs):
        x1 = tf.gather(inputs, self.pairs1, axis=1)
        x2 = tf.gather(inputs, self.pairs2, axis=1)
        return x1 * x2

class Comb2InputError(Exception):
    pass
