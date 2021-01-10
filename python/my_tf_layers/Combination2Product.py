import tensorflow as tf
from tensorflow import keras
import numpy as np


class Combination2Product(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        """Called the first time call() is called. input_shape has the shape of each batch"""
        print("in shape: " + str(input_shape))
        num_inputs = input_shape.as_list()[-1]
        if num_inputs == 1:
            raise Comb2InputError()
        num_outputs = int(0.5 * num_inputs * (num_inputs - 1))  # num_inputs choose 2
        print("num outputs: " + str(num_outputs))
        w_init = tf.random_uniform_initializer()
        self.w = tf.Variable(w_init((num_outputs,)), trainable=True)
        print(type(self.w))

    def compile_inputs(self, inputs: tf.Tensor):
        """Return two tensors s.t they represent all possible pairs, element-wise"""
        to_return = [[], []]
        node1 = 0
        node2 = 1
        count = 0  #DEBUG
        while node1 < len(inputs) - 1:
            count += 1
            to_return[0].append(inputs[node1])
            to_return[1].append(inputs[node2])
            node2 += 1
            if node2 == len(inputs):
                node1 += 1
                node2 = node1 + 1
        print("compile combination count: " + str(count))  #DEBUG
        return tf.convert_to_tensor(to_return[0]), tf.convert_to_tensor(to_return[1])

    def call(self, inputs, **kwargs):
        print("call inputs: " + str(len(inputs)))
        a, b = self.compile_inputs(inputs)
        print("a,b len: " + str(len(a)))
        print("weights len: " + str(len(self.w)))
        return self.w * a * b


class Comb2InputError(Exception):
    pass
