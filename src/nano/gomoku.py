import numpy as np


class Gomoku:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size))
        self.player = 1

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.player = 1
        self.gameOver = None

    def next_state(self, move):
        pass
    
    def check_valid_move(self, move):
        pass
    
    def check_game_over(self):
        pass
    
    def check_draw(self):
        pass

    def check_winner(self):
        pass