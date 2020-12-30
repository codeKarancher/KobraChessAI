import chess

def board_evaluator(board: chess.Board, color: chess.Color) -> float:
    return 0

def bfs_best_move(board: chess.Board, color: chess.Color) -> chess.Move:
    return board.generate_legal_moves()[0]