import random
from pathlib import Path

import cv2
from definitions import BOARD_SIZE, GomokuPiece
from pynput.mouse import Listener, Button
from board import *

capture = None
initial_board = None
model = None  # Load model here


def process_image(image_path=Path(__file__) / "debug/temp/maybeBoard.jpeg"):
    if not debug:
        ret, frame = capture.read()
        cv2.imwrite(str(image_path), frame)

    img, gray_blur = get_img_and_blur(image_path)
    edges = canny_edges(gray_blur)

    debug_put_lines_on_image([], gray_blur.copy())

    lines = hough_line(edges)
    debug_put_lines_on_image(lines, gray_blur.copy())
    h_lines, v_lines = h_v_lines(lines)
    intersection_points = line_intersections(h_lines, v_lines)
    points = cluster_points(intersection_points)
    debug_put_points_on_image(points, gray_blur.copy())
    # points = augment_points(points)
    x_list = write_crop_images_2(img, points, 0)
    img_filename_list = grab_cell_files()




def test_detection():
    # process_image(Path(__file__).parent.parent.parent / f"datasets/go_imgs/go_board_{random.randint(1, 140)}.png")
    process_image(Path('/Users/nickrozanov/PycharmProjects/AI_Jetson_Nano/datasets/go_imgs/img/go_board_8.png'))


if __name__ == '__main__':
    debug = True
    if not debug:
        capture = cv2.VideoCapture(0)
    else:
        test_detection()