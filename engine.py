# Copyright <YEAR> <COPYRIGHT HOLDER>
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import chess
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.optimize

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

    def boardToNNRepresentation(self, board: chess.Board):
        array = np.zeros(64)
        piecesDict = board.piece_map()
        for square in piecesDict:
            array[square] = piecesDict.get(square).piece_type
        return array

    def func(self, board: chess.Board):
        input = self.boardToNNRepresentation(board)
        return self.model(np.array(input))

    @classmethod
    def randomModelFromModel(cls, model: keras.Model, deviation=1):
        new_model = keras.models.clone_model(model)
        for layer in new_model.layers:
            layer.set_weights(np.random.uniform(layer.get_weights() - deviation, layer.get_weights() + deviation))
        return Evaluator(new_model)

class ColorError(Exception):
    """Raised if the wrong chess color was detected"""
    pass

class Engine:

    def __init__(self, evaluator: Evaluator, color: chess.Color):
        self.evaluator = evaluator
        self.color = color

    def best_move(self, board: chess.Board):
        if board.turn != self.color:
            raise ColorError

        high = -1000000
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            rating = self.evaluator.func(board)
            if rating > high:
                high = rating
                best_move = move
            board.pop()
        return best_move
