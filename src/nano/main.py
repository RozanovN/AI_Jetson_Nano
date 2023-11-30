import pygame
import keyboard
import numpy as np
import gomoku
from tensorflow.keras.models import load_model

model = load_model("./models/my_model_pad.h5")
board_size = gomoku.size

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
    file_path = "./sound/" + message + ".mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
def main():
    result = None
    while result is None:
        print("Waiting for input...")
        keyboard.wait('esc')
        
        board_state = get_board_state()
        board_state = np.zeros((board_size, board_size))
        result = gomoku.check_game_over(-1, board_state)
        if result == "win":
            announce("blackWin")
        elif result == "draw":
            announce("draw")
        
        row, col = get_model_move(board_state)
        announce(chr(ord('a') + row))
        announce(str(col))
        
        board_state[row][col] = 1
        result = gomoku.check_game_over(1, board_state)
        
        if result == "win":
            announce("whiteWin")
        elif result == "draw":
            announce("draw")
    print("Game over!")

if __name__ == '__main__':
    main()