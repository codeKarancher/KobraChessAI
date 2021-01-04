import chess
from tensorflow import keras
from engine import *

project_path = "/Users/karan/Desktop/KobraChessAI"
model_name = "stockfish_coached_sn1"
input_color = "any"
while input_color != "w" and input_color != "b":
    print("AI colour? (w/b)")
    input_color = input()
if input_color == "w":
    AI_color = chess.WHITE
    user_color = "b"
else:
    AI_color = chess.BLACK
    user_color = "w"
color_dict = {"w" : "white", "b" : "black"}
kobra_chess_ai = Engine(Evaluator(keras.models.load_model(project_path + "/Saved_Models/" + model_name)), AI_color)
board = chess.Board()
winner = "ai"
while not board.is_checkmate():
    print(board)
    list_valid_moves = list(board.legal_moves)
    sorted_valid_moves = []
    for valid_move in list_valid_moves:
        sorted_valid_moves = sorted_valid_moves + [valid_move.uci()]
    sorted_valid_moves.sort()
    s = ""
    for move in sorted_valid_moves:
        s = s + move + " "
    print("\n" + color_dict[user_color] + " to move... valid moves:")
    print(s + "\nInput your move")
    user_move = input()
    board.push_uci(user_move)
    if board.is_checkmate():
        winner = "user"
        break
    print(board)
    print("\n" + color_dict[input_color] + " to move; KobraChessAI is thinking...")
    ai_move = kobra_chess_ai.best_move(board)
    print("KobraChessAI decided to play " + ai_move.uci())
    board.push(ai_move)

if winner == "user":
    print('Checkmate! You won!')
else:
    print('Checkmate! Oh well, you know what they say about Kobras...')
