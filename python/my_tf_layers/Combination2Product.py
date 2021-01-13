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
        num_outputs = int(0.5 * num_inputs * (num_inputs - 1))  # num_inputs choose 2
        w_init = tf.random_uniform_initializer()
        self.w = tf.Variable(w_init((num_outputs,)), trainable=True)

    def eval_input(self, inputs: tf.Tensor):
        """Returns a tensor containing products of all unique pairings of values in 'inputs' tensor,
        multiplied through by the trainable weights (1 per pair)"""
        len_inputs = len(inputs)
        a = []
        b = []

        def register_new_pair(node1, node2):
            a.append(inputs[node1])
            b.append(inputs[node2])
            node2 += 1
            if node2 == len_inputs:
                node1 += 1
                node2 = node1 + 1
            return node1, node2
        # Must convert a and b to tensors...
        tf.while_loop(cond=lambda n1, n2: n1 < len_inputs, body=register_new_pair, loop_vars=(0, 1))
        return self.w * tf.convert_to_tensor(a) * tf.convert_to_tensor(b)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self.eval_input, elems=inputs)


class Comb2InputError(Exception):
    pass
