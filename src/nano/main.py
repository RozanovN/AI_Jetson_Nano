import numpy as np
from gomoku import Gomoku
from tensorflow.keras.models import load_model

model1 = load_model("./models/20201213_202430.h5")
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

def main():
    gomoku = Gomoku(board_size)
    current_player = 1
    while not gomoku.game_over:
        if players[current_player] is not "human":
            move = get_model_move(players[current_player], gomoku.board * current_player, gomoku)
        else:
            move = get_human_move(gomoku)
        gomoku.next_state(move)
        if gomoku.check_win():
            print("Player %d wins!" % 1 if current_player == 1 else 2)
        elif gomoku.check_draw():
            print("Draw!")
        current_player = -current_player
        print(gomoku.board)
        print("==================================================================")  
    print("Game over!")

if __name__ == '__main__':
    main()