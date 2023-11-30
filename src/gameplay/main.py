import numpy as np
from gomoku import Gomoku
from tensorflow.keras.models import load_model
import tkinter as tk
import time

model1 = load_model("./models/my_model_pad.h5")
model2 = load_model("./models/20201213_202430.h5")
board_size = 20
players = {1: model1, -1: model2}

def get_model_move(model, board_state, board):
    output = model.predict(np.array([board_state]))
    move = None
    while move is None or not board.check_valid_move(move):
        if move is not None:
            print(f"Invalid move {move}. Getting next move...")
        predicted_index = np.argmax(output)
        output[0][predicted_index] = 0
        row = predicted_index // board_size
        col = predicted_index % board_size
        move = (row, col)
    return move

def get_human_move(board):
    move = None
    while move is None or not board.check_valid_move(move):
        if move is not None:
            print(f"Invalid move {move}. Try again...")
        row = int(input(f"Enter row (0-{board_size-1}): "))
        col = int(input(f"Enter col (0-{board_size-1}): "))
        move = (row, col)
    return move

class GomokuBoardDisplay:
    def __init__(self, root, gomoku):
        self.root = root
        self.gomoku = gomoku
        self.canvas = tk.Canvas(root, width=400, height=400, bg="gray")
        self.canvas.pack()
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(len(self.gomoku.board)):
            for col in range(len(self.gomoku.board[row])):
                x1, y1 = col * 20, row * 20
                x2, y2 = x1 + 20, y1 + 20
                stone_color = "black" if self.gomoku.board[row, col] == 1 else "white" if self.gomoku.board[row, col] == -1 else "gray"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=stone_color)

def main():
    gomoku = Gomoku(board_size)

    root = tk.Tk()
    root.title("Gomoku Board Display")

    board_display = GomokuBoardDisplay(root, gomoku)

    current_player = 1
    while not gomoku.game_over:
        if players[current_player] != "human":
            move = get_model_move(players[current_player], gomoku.board * current_player, gomoku)
        else:
            move = get_human_move(gomoku)
        gomoku.next_state(move)
        board_display.draw_board()
        root.update()
        time.sleep(0.5)
        if gomoku.check_win():
            print(gomoku.board)
            print("Player %d won!" % (1 if current_player == 1 else 2))
        elif gomoku.check_draw():
            print("Draw!")
        current_player = -current_player
        gomoku.change_player()

    print("Game over!")

    root.mainloop()

if __name__ == '__main__':
    main()