import os
import re
import time
import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model


raw_data_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "gameplay", "raw")
pattern =  re.compile(r'\b\d+,\d+,\d+\b')
size_of_board = 20
initial_board_state = np.zeros((size_of_board, size_of_board), dtype=int)
raw_data_file_extension = ".psq"
dataset = []

def file_generator(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".psq"):
            yield os.path.join(folder_path, filename)

def on_click(event):
    try:
        last_line = None
        file_path = next(file_gen)
        canvas.delete("ovals")
        with open(file_path, 'r') as file:
            for line in file:
                if re.search(pattern, line):
                        last_line = line
                        create_oval(line)
                        root.update_idletasks()  # Update the Tkinter window
                        root.after(200)
            create_oval(last_line, last=True)
            root.update_idletasks()
    except StopIteration:
        print("No more files in the folder.")
        
def create_oval(line, last=False):
    global fill
    if fill == "black":
        fill = "white"
    else:
        fill = "black"
    if last:
        fill = "red"
    tokens = line.strip().split(",")
    row = int(tokens[0]) - 1
    col = int(tokens[1]) - 1
    canvas.create_oval(
        col * cell_size + board_left - oval_offset,
        row * cell_size + board_top - oval_offset,
        (col + 1) * cell_size + board_left - oval_offset,
        (row + 1) * cell_size + board_top - oval_offset,
        fill=fill,
        tags="ovals"
    )
        
file_gen = file_generator(raw_data_dir_path)
fill = "black"

# Set up the main window
root = tk.Tk()
root.title("Gomoku")

# Set the cell size and board size
cell_size = 30
board_size = 20

# Oval offset
oval_offset = cell_size / 2

# Set the size of the canvas
canvas_width = board_size * cell_size * 1.5
canvas_height = board_size * cell_size * 1.5

# Create a canvas
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="grey")
canvas.pack()

# Calculate the offset to center the grid
x_offset = (canvas_width - board_size * cell_size) / 2
y_offset = (canvas_height - board_size * cell_size) / 2

# Calculate board boundaries
board_left = x_offset
board_right = (board_size - 1) * cell_size + x_offset
board_top = y_offset
board_bottom = (board_size - 1) * cell_size + y_offset

# Setup initial board state
board = np.zeros((board_size, board_size))
game_over = False

# Draw the grid lines
for i in range(board_size):
    canvas.create_line(
        board_left + i * cell_size, board_top, board_left + i * cell_size, board_bottom
    )  # vertical
    canvas.create_line(
        board_left, board_top + i * cell_size, board_right, board_top + i * cell_size
    )  # horizontal

# Bind the click event
canvas.bind("<Button-1>", on_click)

# Run the Tkinter event loop
root.mainloop()
