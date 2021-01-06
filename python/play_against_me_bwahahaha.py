import PySimpleGUI as sg
import os
import sys
import chess
import copy
from python.engine import *
import tensorflow as tf
from tensorflow import keras

PROJECT_PATH = "/Users/karan/Desktop/KobraChessAI"
CHESS_PATH = PROJECT_PATH + "/python/Assets"  # path to the chess pieces

BLANK = 0  # piece names
PAWNB = 1
KNIGHTB = 2
BISHOPB = 3
ROOKB = 4
QUEENB = 5
KINGB = 6
PAWNW = 7
KNIGHTW = 8
BISHOPW = 9
ROOKW = 10
QUEENW = 11
KINGW = 12


def chess_piece_to_piece_name(piece: chess.Piece):
    if piece is None:
        return BLANK
    if piece.color == chess.BLACK:
        piece_name = piece.piece_type
    else:
        piece_name = piece.piece_type + 6
    return piece_name


initial_board = [[ROOKB, KNIGHTB, BISHOPB, QUEENB, KINGB, BISHOPB, KNIGHTB, ROOKB],
                 [PAWNB, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [PAWNW, ] * 8,
                 [ROOKW, KNIGHTW, BISHOPW, QUEENW, KINGW, BISHOPW, KNIGHTW, ROOKW]]

blank = os.path.join(CHESS_PATH, 'blank.png')
bishopB = os.path.join(CHESS_PATH, 'bishopb.png')
bishopW = os.path.join(CHESS_PATH, 'bishopw.png')
pawnB = os.path.join(CHESS_PATH, 'pawnb.png')
pawnW = os.path.join(CHESS_PATH, 'pawnw.png')
knightB = os.path.join(CHESS_PATH, 'knightb.png')
knightW = os.path.join(CHESS_PATH, 'knightw.png')
rookB = os.path.join(CHESS_PATH, 'rookb.png')
rookW = os.path.join(CHESS_PATH, 'rookw.png')
queenB = os.path.join(CHESS_PATH, 'queenb.png')
queenW = os.path.join(CHESS_PATH, 'queenw.png')
kingB = os.path.join(CHESS_PATH, 'kingb.png')
kingW = os.path.join(CHESS_PATH, 'kingw.png')

images = {BISHOPB: bishopB, BISHOPW: bishopW, PAWNB: pawnB, PAWNW: pawnW, KNIGHTB: knightB, KNIGHTW: knightW,
          ROOKB: rookB, ROOKW: rookW, KINGB: kingB, KINGW: kingW, QUEENB: queenB, QUEENW: queenW, BLANK: blank}


def open_pgn_file(filename):
    pgn = open(filename)
    first_game = chess.pgn.read_game(pgn)
    moves = [move for move in first_game.main_line()]
    return moves


def render_square(image, key, location):
    if (location[0] + location[1]) % 2:
        color = '#B58863'
    else:
        color = '#F0D9B5'
    return sg.RButton('', image_filename=image, size=(1, 1), button_color=('white', color), pad=(0, 0), key=key)


def redraw_board(window, board):
    for i in range(8):
        for j in range(8):
            color = '#B58863' if (i + j) % 2 else '#F0D9B5'
            piece_image = images[board[i][j]]
            elem = window.FindElement(key=(i, j))
            elem.Update(button_color=('white', color),
                        image_filename=piece_image, )


def to_psg_board(board: chess.Board):
    psg_board = [[], [], [], [], [], [], [], []]
    for i in range(8):
        for j in range(8):
            psg_board[i].append(chess_piece_to_piece_name(board.piece_at(8 * (7-i) + j)))
    return psg_board


def playGame():
    menu_def = [['&File', ['&Open PGN File', 'E&xit']],
                ['&Help', '&About...'], ]

    # sg.SetOptions(margins=(0,0))
    sg.ChangeLookAndFeel('GreenTan')
    # create initial board setup
    board = chess.Board()
    psg_board = to_psg_board(board)
    # the main board display layout
    board_layout = [[sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh']]
    # loop though board and create buttons with images
    for i in range(8):
        row = [sg.T(str(8 - i) + '   ', font='Any 13')]
        for j in range(8):
            piece_image = images[psg_board[i][j]]
            row.append(render_square(piece_image, key=(i, j), location=(i, j)))
        row.append(sg.T(str(8 - i) + '   ', font='Any 13'))
        board_layout.append(row)
    # add the labels across bottom of board
    board_layout.append([sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh'])

    # setup the controls on the right side of screen
    openings = (
        'Any', 'Defense', 'Attack', 'Trap', 'Gambit', 'Counter', 'Sicillian', 'English', 'French', 'Queen\'s openings',
        'King\'s Openings', 'Indian Openings')

    board_controls = [[sg.RButton('New Game', key='New Game'), sg.RButton('Draw')],
                      [sg.RButton('Resign Game'), sg.RButton('Set FEN')],
                      [sg.RButton('Player Odds'), sg.RButton('Training')],
                      [sg.Drop(openings), sg.Text('Opening/Style')],
                      [sg.CBox('Play As White', key='_white_')],
                      [sg.Drop([2, 3, 4, 5, 6, 7, 8, 9, 10], size=(3, 1), key='_level_'), sg.Text('Difficulty Level')],
                      [sg.Text('Move List')],
                      [sg.Multiline([], do_not_clear=True, autoscroll=True, size=(15, 10), key='_movelist_')],
                      ]

    # layouts for the tabs
    controls_layout = [[sg.Text('Performance Parameters', font='_ 20')],
                       [sg.T('Put stuff like AI engine tuning parms on this tab')]]

    statistics_layout = [[sg.Text('Statistics', font='_ 20')],
                         [sg.T('Game statistics go here?')]]

    board_tab = [[sg.Column(board_layout)]]

    # the main window layout
    layout = [[sg.Menu(menu_def, tearoff=False)],
              [sg.TabGroup([[sg.Tab('Board', board_tab),
                             sg.Tab('Controls', controls_layout),
                             sg.Tab('Statistics', statistics_layout)]], title_color='red'),
               sg.Column(board_controls)],
              [sg.Text('Click anywhere on board for next move', font='_ 14')]]

    window = sg.Window('Chess',
                       default_button_element_size=(12, 1),
                       auto_size_buttons=False,
                       icon='kingb.ico').Layout(layout)

    kobra_chess_ai_model = keras.models.load_model(PROJECT_PATH + "/Saved_Models/stockfish_coached_sn3")
    kobra_chess_ai = Engine(Evaluator_Type3(kobra_chess_ai_model), color=chess.BLACK)

    move_count = 1
    move_state = move_from = move_to = 0
    # ---===--- Loop taking in user input --- #
    while not board.is_game_over():

        if board.turn == chess.WHITE:
            # human_player(board)
            move_state = 0
            while True:
                button, value = window.Read()
                if button in (None, 'Exit'):
                    exit()
                if button == 'New Game':
                    board = chess.Board()
                    psg_board = to_psg_board(board)
                    redraw_board(window, psg_board)
                    move_state = 0
                    break
                level = value['_level_']
                if type(button) is tuple:
                    if move_state == 0:
                        move_from = button
                        row, col = move_from
                        piece = psg_board[row][col]  # get the move-from piece
                        button_square = window.FindElement(key=(row, col))
                        button_square.Update(button_color=('white', 'red'))
                        move_state = 1
                    elif move_state == 1:
                        move_to = button
                        row, col = move_to
                        if move_to == move_from:  # cancelled move
                            color = '#B58863' if (row + col) % 2 else '#F0D9B5'
                            button_square.Update(button_color=('white', color))
                            move_state = 0
                            continue

                        picked_move = '{}{}{}{}'.format('abcdefgh'[move_from[1]], 8 - move_from[0],
                                                        'abcdefgh'[move_to[1]], 8 - move_to[0])
                        print(picked_move) #DEBUG
                        print(board) #DEBUG

                        if picked_move in [str(move) for move in board.legal_moves]:
                            board.push(chess.Move.from_uci(picked_move))
                            print(board) #DEBUG
                        else:
                            print('Illegal move')
                            move_state = 0
                            color = '#B58863' if (move_from[0] + move_from[1]) % 2 else '#F0D9B5'
                            button_square.Update(button_color=('white', color))
                            continue

                        psg_board = to_psg_board(board)
                        redraw_board(window, psg_board)
                        move_count += 1

                        window.FindElement('_movelist_').Update(picked_move + '\n', append=True)

                        break
        else:
            best_move = kobra_chess_ai.best_move(board)
            move_str = str(best_move)

            window.FindElement('_movelist_').Update(move_str + '\n', append=True)

            board.push(best_move)
            psg_board = to_psg_board(board)
            redraw_board(window, psg_board)

            move_count += 1
    sg.Popup('Game over!', 'Thank you for playing')


playGame()
