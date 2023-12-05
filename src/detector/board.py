import glob
import re
from collections import defaultdict
from itertools import accumulate
from locale import atoi
from pathlib import Path

import cv2
import math
import numpy as np
import scipy.cluster as cluster
import scipy.spatial as spatial
from PIL import Image

from definitions import BOARD_LENGTH


def get_img_and_blur(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (483, 483))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = gray #  cv2.blur(gray, (1, 1))
    return img, gray_blur


def canny_edges(img, sigma=0.25):
    v = np.median(img)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.GaussianBlur(img, (5, 5), 2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def hough_line(edges):
    rho = 0.5
    theta = np.pi / 180
    threshold = 120
    min_line_length = 100
    max_line_gap = 5
    lines = cv2.HoughLines(edges, rho, theta, threshold, None, min_line_length, max_line_gap)
    lines = unique_lines(lines)
    return np.reshape(lines, (-1, 2))


def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta, in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


def unique_lines(lines, delta=15):
    v = accumulate(lines, lambda x, y: x if abs(y[0][0] - x[0][0]) < delta else y)
    l = [i for i in v]

    if l is None or len(l) == 0:
        return None
    else:
        return np.unique(np.array(l), axis=0)


def debug_put_lines_on_image(lines, img, folder_path=str(Path(__file__).parent / 'debug/lines/')):
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

    cv2.imwrite(folder_path + '/result.jpg', img)


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    #  t - the maximum inter-cluster distance allowed:
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = np.mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


def write_crop_images(img, points, img_count=0, folder_path=str(Path(__file__).parent / 'debug/raw_data/')):
    num_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= BOARD_LENGTH:
        range_num = BOARD_LENGTH
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - BOARD_LENGTH) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            try:
                cv2.imwrite(folder_path + '/' + str(img_count) + '.jpeg', cropped)
                img_count += 1
            except Exception as e:
                print(e)
    return img_count


def write_crop_images_2(img, points, img_count=0, folder_path=str(Path(__file__).parent / 'debug/raw_data/')):
    length = float('inf')  # Initialize with a large value

    for i in range(len(points) - 1):
        distance = np.linalg.norm(np.asarray(points[i]) - np.asarray(points[i + 1]))
        if length > distance:
            length = distance

    length = int(length) + 2
    start_point = points[0]
    start_x = int(start_point[0] - length // 2)
    start_y = int(start_point[1] - length // 2)

    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    copy_start_x = start_x
    for row in range(BOARD_LENGTH):
        for column in range(BOARD_LENGTH):
            cropped = img[start_y: start_y + length, start_x: start_x + length]
            try:
                cv2.imwrite(folder_path + '/data_image' + str(img_count) + '.jpeg', cropped)
                img_count += 1
            except Exception as e:
                print(e)
            start_x += length
        start_x = copy_start_x
        start_y += length

    return img_count


def grab_cell_files(folder_path=str(Path(__file__).parent / 'debug/raw_data/*')):
    img_filename_list = []
    for path_name in glob.glob(folder_path):
        img_filename_list.append(path_name)
    # img_filename_list = img_filename_list.sort(key=natural_keys)
    return img_filename_list


def debug_put_points_on_image(points, img, folder_path=str(Path(__file__).parent / 'debug/points/')):
    for point in points:
        img = cv2.circle(img, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 255), thickness=1)

    cv2.imwrite(folder_path + '/result.jpg', img)


def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data


def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text) if c.isnumeric()]


def classify_cells(model, img_filename_list):
    #
    category_reference = {0: -1, 1: 0, 2: 1}
    pred_list = []
    img_filename_list.sort(key=natural_keys)
    for filename in img_filename_list:
        img = prepare_image(filename)
        out = model.predict(img)
        top_pred = np.argmax(out)
        pred = category_reference[top_pred]
        pred_list.append(pred)

    board = []
    for row in range(BOARD_LENGTH):
        entry = []
        for column in range(BOARD_LENGTH):
            entry.append(pred_list[column + row * 19])
        board.append(entry)
    return board


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


def perspective_transformation(img, gray):
    img_original = img.copy()
    edged = cv2.Canny(gray, 10, 20)

    # Contour detection
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

    cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    # max_height = max(int(right_height), int(left_height))
    max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

    # Image shape modification for hstack
    gray = np.stack((gray,) * 3, axis=-1)
    edged = np.stack((edged,) * 3, axis=-1)

    img_hor = np.hstack((img_original, gray, edged, img))
    cv2.imshow("Contour detection", img_hor)
    cv2.imshow("Warped perspective", img_output)

    # cv2.imwrite('output/document.jpg', img_output)

    cv2.waitKey(0)