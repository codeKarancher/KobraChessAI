import tensorflow as tf
from tensorflow import keras


class Combination2(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.combination_map = {}
        self.kernel = tf.constant([])

    def build(self, input_shape):
        num_outputs = int(0.5 * input_shape[-1] * (input_shape[-1] - 1))  # input_shape[-1] choose 2
        node1 = 0
        node2 = 1
        for out_node in range(num_outputs):
            self.combination_map[out_node] = [node1, node2]
            node2 += 1
            if node2 == input_shape[-1]:
                node1 += 1
                node2 = node1 + 1
        self.kernel = self.add_weight('kernel', shape=2 * num_outputs, trainable=True)
        return

    def call(self, inputs, **kwargs) -> tf.Tensor:
        values = []
        for out_node in range(len(self.combination_map)):
            in_nodes = self.combination_map.get(out_node)
            values.append(self.kernel[out_node*2]*inputs[in_nodes[0]] + self.kernel[out_node*2+1]*inputs[in_nodes[1]])
        return tf.Tensor(values)
