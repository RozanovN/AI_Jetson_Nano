import os
import random
import pygame
import keyboard
import numpy as np
import gomoku
from tensorflow.keras.models import load_model

sound_dir = os.path.join(os.path.dirname(__file__), "sound")
model_dir = os.path.join(os.path.dirname(__file__), "models")
model = load_model(os.path.join(model_dir, "my_model_pad_tanh.h5"))
board_size = gomoku.board_size

def get_model_move(board_state):
    output = model.predict(np.array([board_state]))
    move = None
    while move is None or not gomoku.check_valid_move(board_state, move):
        if move is not None:
            print(f"Invalid move {move}. Getting next move...")
        predicted_index = np.argmax(output)
        output[0][predicted_index] = 0
        row = predicted_index // board_size
        col = predicted_index % board_size
        move = (row, col)
    return move

def get_board_state():
    state = None
    # TODO: Image recognition, get board state with camera
    return state

def announce(message):
    file_path = os.path.join(sound_dir, message + ".mp3")
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
def main():
    result = None
    board_state = np.zeros((board_size, board_size))
    player = -1
    while result is None:
        if player == -1:
            print("Waiting for input...")
            keyboard.wait('esc')
            # board_state = get_board_state()
            board_state[random.randint(0, board_size - 1)][random.randint(0, board_size - 1)] = player
        else:
            row, col = get_model_move(board_state)
            announce(chr(ord('a') + row))
            announce(str(col))
            board_state[row][col] = player
        result = gomoku.check_game_over(player, board_state)
        if result == "win":
            print(board_state)
            if player == -1:
                announce("blackWin")
            else:
                announce("whiteWin")
            break
        elif result == "draw":
            announce("draw")
            break
        player *= -1
    print("Game over!")

if __name__ == '__main__':
    main()