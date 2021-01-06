import chess
import tensorflow
from tensorflow import keras
from python.engine import *

PROJECT_PATH = "/Users/karan/Desktop/KobraChessAI"


def match(uid_1, uid_2, num_matches_per_colour=10) -> int:
    if uid_1 < 3:
        evaluator1 = Evaluator_Type1_deprecated(keras.models.load_model(PROJECT_PATH + "/Saved_Models/" + str(uid_1)))
    else:
        evaluator1 = Evaluator_Type3(keras.models.load_model(PROJECT_PATH + "/Saved_Models/" + str(uid_1)))
    if uid_2 < 3:
        evaluator2 = Evaluator_Type1_deprecated(keras.models.load_model(PROJECT_PATH + "/Saved_Models/" + str(uid_2)))
    else:
        evaluator2 = Evaluator_Type3(keras.models.load_model(PROJECT_PATH + "/Saved_Models/" + str(uid_2)))
    to_return = 0
    for i in range(num_matches_per_colour):
        to_return += game(Engine(evaluator1, color=chess.WHITE), Engine(evaluator2, color=chess.BLACK))\
                     - game(Engine(evaluator2, color=chess.WHITE), Engine(evaluator1, color=chess.BLACK))
    return 0  # DEBUG


def game(white: Engine, black: Engine) -> int:
    board = chess.Board()
    current_winner = False
    current_player = {False: white, True: black}
    while not board.is_game_over():
        board.push(current_player[current_winner].best_move(board))
        current_winner = not current_winner
    return 46 if current_winner else -54
