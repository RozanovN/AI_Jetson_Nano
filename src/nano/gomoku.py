import numpy as np


class Gomoku:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size))
        self.player = 1

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.player = 1

    def next_state(self, move):
        if not self.check_valid_move(move):
            raise ValueError("Invalid move")
        self.player = -self.player
        self.board[move[0]][move[1]] = self.player
    
    def check_valid_move(self, move):
        row = move[0]
        col = move[1]
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if self.board[row][col] != 0:
            return False
        return True
    
    def check_draw(self):
        if np.count_nonzero(self.board) == self.size * self.size:
            return True
        return False

    def check_win(self):
        for row in range(self.size):
            for col in range(self.size):
                try:
                    if (
                        self.board[row][col] == self.player
                        and self.board[row + 1][col] == self.player
                        and self.board[row + 2][col] == self.player
                        and self.board[row + 3][col] == self.player
                        and self.board[row + 4][col] == self.player
                    ):
                        return True
                except:
                    pass
                try:
                    if (
                        self.board[row][col] == self.player
                        and self.board[row][col + 1] == self.player
                        and self.board[row][col + 2] == self.player
                        and self.board[row][col + 3] == self.player
                        and self.board[row][col + 4] == self.player
                    ):
                        return True
                except:
                    pass
                try:
                    if (
                        self.board[row][col] == self.player
                        and self.board[row + 1][col + 1] == self.player
                        and self.board[row + 2][col + 2] == self.player
                        and self.board[row + 3][col + 3] == self.player
                        and self.board[row + 4][col + 4] == self.player
                    ):
                        return True
                except:
                    pass
                try:
                    if (
                        col >= 4
                        and self.board[row][col] == self.player
                        and self.board[row + 1][col - 1] == self.player
                        and self.board[row + 2][col - 2] == self.player
                        and self.board[row + 3][col - 3] == self.player
                        and self.board[row + 4][col - 4] == self.player
                    ):
                        return True
                except:
                    pass
        return False