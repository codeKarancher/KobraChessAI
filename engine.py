import chess
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.optimize

def board_evaluator(board: chess.Board, color: chess.Color) -> float:
    return 0

def bfs_best_move(board: chess.Board, color: chess.Color) -> chess.Move:
    return board.generate_legal_moves()[0]

class Evaluator:

    def __init__(self, num_inputs: int, hidden_layer_sizes=[32, 32]):
        input = keras.Input(shape=(num_inputs,))
        x = input
        for i in range(len(hidden_layer_sizes)):
            x = keras.layers.Dense(hidden_layer_sizes[i], activation="relu")(x)
        output = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=input, outputs=output)

    def __init__(self, model: keras.Model):
        self.model = model

    def func(self, input):
        return self.model(np.array([input]))

    @classmethod
    def randomModelFromModel(cls, model: keras.Model, deviation=0.1):
        new_model = keras.models.clone_model(model)
        new_model.trainabl