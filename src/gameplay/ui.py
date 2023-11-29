import tkinter as tk

def on_click(event):
    col = round((event.x - board_left) / cell_size)
    row = round((event.y - board_top) / cell_size)

    # Draw the piece
    if 0 <= col < board_size and 0 <= row < board_size:
        canvas.create_oval(col * cell_size + board_left - oval_offset, row * cell_size + board_top - oval_offset,
                            (col + 1) * cell_size + board_left - oval_offset, (row + 1) * cell_size + board_top - oval_offset, fill="black")

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
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Calculate the offset to center the grid
x_offset = (canvas_width - board_size * cell_size) / 2
y_offset = (canvas_height - board_size * cell_size) / 2

# Calculate board boundaries
board_left = x_offset
board_right = (board_size - 1) * cell_size + x_offset
board_top = y_offset
board_bottom = (board_size - 1) * cell_size + y_offset

# Draw the grid lines
for i in range(board_size):
    canvas.create_line(board_left + i * cell_size, board_top, board_left + i * cell_size, board_bottom) # vertical
    canvas.create_line(board_left, board_top + i * cell_size, board_right, board_top + i * cell_size) # horizontal

# Bind the click event
canvas.bind("<Button-1>", on_click)

# Run the Tkinter event loop
root.mainloop()
