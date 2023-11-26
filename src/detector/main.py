
import cv2
from definitions import BOARD_SIZE, GomokuPiece
from pynput.mouse import Listener, Button
from board import *

capture = None
initial_board = None
model = None  # Load model here


def process_image():
    ret, frame = capture.read()

    cv2.imwrite('maybeBoard.jpeg', frame)
    img, gray_blur = get_img_and_blur('maybeBoard.jpeg')
    edges = canny_edges(gray_blur)
    lines = hough_line(edges)
    h_lines, v_lines = h_v_lines(lines)
    intersection_points = line_intersections(h_lines, v_lines)


def on_move(x, y, button, pressed):
    if button == Button.left and pressed:
        process_image()
    if button == Button.middle and pressed:
        return False


def main():
    #  start video capturing
    listener = Listener(on_click=on_move)
    listener.start()


if __name__ == '__main__':
    capture = cv2.VideoCapture(1)
    initial_board = [GomokuPiece.NP for _ in range(BOARD_SIZE + 1)]
    main()