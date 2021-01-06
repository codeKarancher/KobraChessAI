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
import numpy as np
import tensorflow as tf
from tensorflow import keras
import stockfish
from python.engine import *


class MyStockfishBoardEvaluator(stockfish.Stockfish):

    def __init__(self, depth=10):
        super().__init__(path="/usr/local/Cellar/stockfish/12/bin/stockfish", depth=depth)
        self.int_depth = depth

    def get_evaluation_depth_limited(self) -> dict:
        """Evaluates current position

        Returns:
            A dictionary of the current advantage with "type" as "cp" (centipawns) or "mate" (checkmate in)
        """

        evaluation = dict()
        fen = self.get_fen_position()
        if "w" in fen:  # w can only be in FEN if it is whites move
            compare = 1
        else:  # stockfish shows advantage relative to current player, convention is to do white positive
            compare = -1
        self._put("position " + fen + "\n go")
        curdepth = 0
        while curdepth <= self.int_depth:
            text = self._read_line()
            splitted_text = text.split(" ")
            if "depth" in splitted_text:
                curdepth = int(splitted_text[splitted_text.index("depth") + 1])
            if splitted_text[0] == "info" and "score" in splitted_text:
                n = splitted_text.index("score")
                evaluation = {"type": splitted_text[n + 1],
                              "value": int(splitted_text[n + 2]) * compare}
            elif splitted_text[0] == "bestmove":
                return evaluation
        return evaluation


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_IDs=np.arange(16384), batch_size=32, move_range=np.arange(5, 125), dim=(320,),
                 input_dtype=int, stockfish_depth=10, shuffle=True):
        self.data_IDs = data_IDs
        self.batch_size = batch_size
        self.move_range = move_range
        self.dim = dim
        self.input_dtype = input_dtype
        self.stockfish = None
        self.stockfish_depth = stockfish_depth
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data_IDs) // self.batch_size  # extra data_IDs will be left out

    def __getitem__(self, index):
        board_seeds = self.data_IDs[index * self.batch_size:(index + 1) * self.batch_size]

        def random_board(num_moves) -> chess.Board:
            board = chess.Board()
            for i in range(num_moves):
                moves = list(board.legal_moves)
                if len(moves) == 0:
                    i -= 2
                    continue
                next_stale_mate = True
                if i < 19:  # Impossible to have stalemate before 19 moves are played (found this online)
                    # The 19 move game(s) is also avoidable, so does not trigger 'next_stale_mate'
                    next_stale_mate = False
                else:
                    for j in range(len(moves)):
                        board.push(moves[j])
                        num_moves = len(list(board.legal_moves))
                        board.pop()
                        if num_moves != 0:
                            next_stale_mate = False
                            break
                board.push(moves[np.random.choice(np.arange(len(moves)))])
                if next_stale_mate:
                    return board
            return board

        def init_stockfish(board: chess.Board):
            if self.stockfish is not None:
                self.stockfish.__del__()
            self.stockfish = MyStockfishBoardEvaluator(depth=self.stockfish_depth)
            self.stockfish.set_fen_position(board.fen())

        x = np.empty((self.batch_size, *self.dim), dtype=self.input_dtype)
        y = np.empty(self.batch_size, dtype=int)

        for i, seed in enumerate(board_seeds):
            np.random.seed(seed)
            seeded_board = random_board(self.move_range[seed % len(self.move_range)])
            x[i,] = boardToOneHotNNInput(seeded_board)
            init_stockfish(seeded_board)
            evaluation = self.stockfish.get_evaluation_depth_limited()
            y[i] = evaluation['value'] if evaluation['type'] == 'cp' else 1000000000
            # if mate in n moves, board is valued extremely highly ~ 2^30 ~ 1,000,000,000

        return (x, y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_IDs)


def coaching_sn3():
    training_generator = DataGenerator(data_IDs=np.arange(16384))
    validation_generator = DataGenerator(data_IDs=np.arange(16384, 16384 + 4096))  # 4096 validation cases

    input = keras.Input(shape=(320,), batch_size=32)
    x = input
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    model.fit(training_generator, validation_data=validation_generator, use_multiprocessing=True, workers=6,
              verbose=True,
              epochs=3)
    model.save("/Users/karan/Desktop/KobraChessAI/Saved_Models/stockfish_coached_sn3")


coaching_sn3()


# -------------------------------------------------- DEPRECATED CODE -------------------------------------------------


class DataGenerator_deprecated(keras.utils.Sequence):

    def __init__(self, seeds=np.arange(0, 16384), list_n_moves=np.arange(25, 125), batch_size=32, dim=(64,),
                 shuffle=True, stockfish_depth=10):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.seeds = seeds
        self.shuffle = shuffle
        self.list_n_moves = list_n_moves
        self.stockfish_depth = stockfish_depth
        self.init_stockfish()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.seeds) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs (seeds)
        seeds_temp = [self.seeds[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(seeds_temp)
        return (x, y)

    def init_stockfish(self):
        self.stockfish = MyStockfishBoardEvaluator(self.stockfish_depth)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.seeds))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, seeds_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, seedID in enumerate(seeds_temp):
            board = self.__random_board(seedID)
            x[i,] = boardToNNInput_deprecated(board)
            self.stockfish.__del__()
            self.init_stockfish()
            self.stockfish.set_fen_position(board.fen())
            eval = self.stockfish.get_evaluation_depth_limited()
            if eval["type"] == "cp":
                y[i] = eval["value"]
            else:  # mate
                y[i] = (10 - eval["value"]) * 200
        return x, y

    def __random_board(self, ID):
        num_moves = self.list_n_moves[ID % len(self.list_n_moves)]
        np.random.seed(self.seeds[ID % len(self.seeds)])
        choices = list(np.random.choice(np.arange(1000), size=num_moves))
        board = chess.Board()
        backtrack_offset = 0
        for i in range(num_moves):
            moves = list(board.legal_moves)
            len_moves = len(moves)
            if len_moves == 0:
                board.pop()
                i -= 2
                backtrack_offset = -1
                continue
            board.push(moves[(choices[i] + backtrack_offset) % len_moves])
            backtrack_offset = 0
        return board


def coaching_deprecated():
    training_generator = DataGenerator_deprecated()
    validation_generator = DataGenerator_deprecated(seeds=np.arange(16384, 16384 + 4096))  # 4096 validation cases

    input = keras.Input(shape=(64,), batch_size=32)
    x = input
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    output = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    model.fit(training_generator, validation_data=validation_generator, use_multiprocessing=True, workers=6,
              verbose=True,
              epochs=4)
    model.save("/Users/karan/Desktop/KobraChessAI/Saved_Models/stockfish_coached_sn2")
