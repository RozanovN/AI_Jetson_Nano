import os
import re
import numpy as np
import pandas as pd


raw_data_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "gameplay", "raw")
output_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "gameplay", "processed")
pattern =  re.compile(r'\b\d+,\d+,\d+\b')
size_of_board = 20
initial_board_state = np.zeros((size_of_board, size_of_board), dtype=int)
raw_data_file_extension = ".psq"
file_count = 1000

# loop through all files in raw_data_dir_path
for filename in os.listdir(raw_data_dir_path):
    if not filename.endswith(raw_data_file_extension):
        continue
    if file_count == 0:
        break
    file_count -= 1
    board_state = np.copy(initial_board_state)
    dataset = []
    with open(os.path.join(raw_data_dir_path, filename), "r") as f:
        for line in f:
            if re.search(pattern, line):
                tokens = line.strip().split(",")
                move_row = int(tokens[0]) - 1
                move_col = int(tokens[1]) - 1
                
                board_state = -board_state
                input = np.copy(board_state)
                
                output = np.copy(initial_board_state)
                output[move_row][move_col] = 1
                
                flipped_input = np.flipud(input)
                flipped_output = np.flipud(output)
                
                for input, output in [(input, output), (flipped_input, flipped_output)]:
                    for i in range(4):
                        rotated_input = np.rot90(input, i)
                        rotated_output = np.rot90(output, i)
                        dataset.append({'input': rotated_input, 'output': rotated_output})
                
                board_state[move_row][move_col] = 1
    np.savez(os.path.join(output_dir_path, filename.split(".")[0] + ".npz"), dataset=dataset)