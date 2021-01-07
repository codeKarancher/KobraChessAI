import chess
import tensorflow
from tensorflow import keras
from python.engine import *

PROJECT_PATH = "/Users/karan/Desktop/KobraChessAI"

openings_dict = {
    "King's Gambit": 'rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2',
    "Queen's Gambit": 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2',
    "Ruy Lopez": 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
    "English":  'rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq - 1 2',
    "Sicilian Defense": 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    "French Defense": 'rnbqkbnr/ppp11ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3',
    "Slav Defense": 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3',
    "Scandinavian Defense": 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2',
    "Dutch Defense": 'rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2',
    "King's Indian Defense": 'rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5',
}
openings_fen = list(openings_dict.values())
print(openings_fen)


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
        to_return += game(Engine(evaluator1, color=chess.WHITE), Engine(evaluator2, color=chess.BLACK), openings_fen[i % len(openings_fen)])\
                     - game(Engine(evaluator2, color=chess.WHITE), Engine(evaluator1, color=chess.BLACK), openings_fen[i % len(openings_fen)])
    return to_return


def game(white: Engine, black: Engine, opening_fen) -> int:
    board = chess.Board()
    current_winner = False
    current_player = {False: white, True: black}
    while not board.is_game_over():
        board.push(current_player[current_winner].best_move(board))
        current_winner = not current_winner
    return (46 if current_winner else -54) if board.is_checkmate() else -2
