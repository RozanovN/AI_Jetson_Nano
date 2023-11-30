import numpy as np


size = 20

def check_valid_move(board, move):
    row, col = move
    if row < 0 or row >= board_size or col < 0 or col >= board_size:
        return False
    if board[row][col] == 0:
        return True
    return False

def check_game_over(player, board):
    if check_win(player, board):
        return "win"
    elif check_board_full(board):
        return "draw"
    return None

def check_board_full(board):
    if np.count_nonzero(board) == board_size * board_size:
        return True
    return False

def check_win(player, board):
    for row in range(board_size):
        for col in range(board_size):
            try:
                if (
                    board[row][col] == player
                    and board[row + 1][col] == player
                    and board[row + 2][col] == player
                    and board[row + 3][col] == player
                    and board[row + 4][col] == player
                ):
                    return True
            except:
                pass
            try:
                if (
                    board[row][col] == player
                    and board[row][col + 1] == player
                    and board[row][col + 2] == player
                    and board[row][col + 3] == player
                    and board[row][col + 4] == player
                ):
                    return True
            except:
                pass
            try:
                if (
                    board[row][col] == player
                    and board[row + 1][col + 1] == player
                    and board[row + 2][col + 2] == player
                    and board[row + 3][col + 3] == player
                    and board[row + 4][col + 4] == player
                ):
                    return True
            except:
                pass
            try:
                if (
                    col >= 4
                    and board[row][col] == player
                    and board[row + 1][col - 1] == player
                    and board[row + 2][col - 2] == player
                    and board[row + 3][col - 3] == player
                    and board[row + 4][col - 4] == player
                ):
                    return True
            except:
                pass
    return False