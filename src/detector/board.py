from itertools import accumulate

import cv2
import numpy as np


def get_img_and_blur(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (7, 7))
    return img, gray_blur


def canny_edges(img, sigma=0.25):
    v = np.median(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (7, 7), 2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def hough_line(edges):
    rho = 2
    theta = np.pi / 180
    threshold = 174
    min_line_length = 100
    max_line_gap = 5
    lines = cv2.HoughLines(edges, rho, theta, threshold, min_line_length, max_line_gap)
    return np.reshape(lines, (-1, 2))


def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta, in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    v_lines = unique_lines(v_lines)
    h_lines = unique_lines(h_lines)
    return h_lines, v_lines


def unique_lines(lines, delta=10):
    """Return lines which are far from each other by more than a given distance"""

    v = accumulate(lines, lambda x, y: x if abs(y[0][0] - x[0][0]) < delta else y)
    l = [i for i in v]

    if l is None or len(l) == 0:
        return None
    else:
        return np.unique(np.array(l), axis=0)


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            # lines use rho (distance from 0, 0) and theta (direction) instead x/y coordinates
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])  # 😭
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)

