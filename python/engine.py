# Copyright 2021 Karan Sharma - ks920@cam.ac.uk
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

piece_cp_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}


def boardToOneHotNNInput(board: chess.Board):
    array = np.zeros(320, dtype=int)
    piecesDict = board.piece_map()
    white_map = {
        chess.PAWN: [100, 0, 0, 0, 0],
        chess.KNIGHT: [0, 100, 0, 0, 0],
        chess.BISHOP: [0, 0, 100, 0, 0],
        chess.ROOK: [0, 0, 0, 100, 0],
        chess.QUEEN: [0, 0, 133, 100, 0],
        chess.KING: [0, 0, 0, 0, 100]
    }
    black_map = {
        chess.PAWN: [-100, 0, 0, 0, 0],
        chess.KNIGHT: [0, -100, 0, 0, 0],
        chess.BISHOP: [0, 0, -100, 0, 0],
        chess.ROOK: [0, 0, 0, -100, 0],
        chess.QUEEN: [0, 0, -133, -100, 0],
        chess.KING: [0, 0, 0, 0, -100]
    }
    data_map = {
        chess.WHITE: white_map,
        chess.BLACK: black_map
    }
    for square in piecesDict:
        piece = piecesDict.get(square)
        array[square*5:(square+1)*5] = data_map[piece.color][piece.piece_type]
    return np.array([array])


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

    def func(self, board: chess.Board) -> float:
        return 0.0

    @classmethod
    def randomModelFromModel(cls, model: keras.Model, deviation=1):
        new_model = keras.models.clone_model(model)
        for layer in new_model.layers:
            layer.set_weights(np.random.uniform(layer.get_weights() - deviation, layer.get_weights() + deviation))
        return Evaluator(new_model)


class Evaluator_Type3(Evaluator):

    def func(self, board: chess.Board) -> float:
        return self.model.predict(boardToOneHotNNInput(board))


class ColorError(Exception):
    """Raised if the wrong chess color was detected"""
    pass


class Engine:

    def __init__(self, evaluator: Evaluator, color: chess.Color):
        self.evaluator = evaluator
        self.color = color

    def best_move(self, board: chess.Board) -> chess.Move:
        if board.turn != self.color:
            raise ColorError

        def is_better(x, y):
            if self.color == chess.WHITE:
                return x > y
            else:
                return y > x

        high = None
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            rating = self.evaluator.func(board)
            if high is None or is_better(rating, high):
                high = rating
                best_move = move
            board.pop()
        return best_move


# -------------------------------------------------- DEPRECATED CODE -------------------------------------------------

def boardToNNInput_deprecated(board: chess.Board):
    array = np.zeros(64, dtype=int)
    piecesDict = board.piece_map()
    for square in piecesDict:
        if piecesDict.get(square).color == chess.WHITE:
            array[square] = piece_cp_values[piecesDict.get(square).piece_type]
        else:
            array[square] = -piece_cp_values[piecesDict.get(square).piece_type]
    return np.array([array])


class Evaluator_Type1_deprecated(Evaluator):

    def func(self, board: chess.Board) -> float:
        input = boardToNNInput_deprecated(board)
        print("SHAPE:" + str(input.shape))
        return self.model.predict(input)

