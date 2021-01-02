import chess
import numpy as np
import tensorflow as tf
from tensorflow import keras

def random_board(num_moves):
    board = chess.Board()
    for i in range(num_moves):
        moves = list(board.legal_moves)
        if len(moves) == 0:
            board.pop()
            i -= 1
            continue
        board.push(np.random.choice(moves))
    return board

keras.utils.Sequence