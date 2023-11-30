from board import *
from model import model
from tensorflow.keras.models import load_model

capture = None
model = model(True)
model.load_weights('model_VGG16.h5')


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
    print(classify_cells(model, img_filename_list))



def test_detection():
    # process_image(Path(__file__).parent.parent.parent / f"datasets/go_imgs/go_board_{random.randint(1, 140)}.png")
    process_image(Path('/Users/nickrozanov/PycharmProjects/AI_Jetson_Nano/datasets/go_imgs/img/go_board_8a.png'))


if __name__ == '__main__':
    debug = True
    if not debug:
        capture = cv2.VideoCapture(0)
    else:
        test_detection()