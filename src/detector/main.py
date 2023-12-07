from board import *
from model import model

model = model(True)
model.load_weights(str(Path(__file__).parent / 'model_VGG16.h5'))
debug = True


def process_image(image_path):
    img, gray_blur = get_img_and_blur(cv2.imread(str(image_path)))
    correct_perspective = None
    try:
        correct_perspective = perspective_transformation(img, gray_blur)
    except Exception as e:
        print(e)

    if correct_perspective is not None:
        cv2.imwrite(str(Path(__file__).parent / 'debug/temp/perspective.jpg'), correct_perspective)
        img, gray_blur = get_img_and_blur(correct_perspective)  

    edges = canny_edges(gray_blur)

    lines = hough_line(edges)
    
    h_lines, v_lines = h_v_lines(lines)
    intersection_points = line_intersections(h_lines, v_lines)
    points = cluster_points(intersection_points)
    # points = augment_points(points)
    x_list = write_crop_images_2(img, points, 0)
    img_filename_list = grab_cell_files()
    classified = classify_cells(model, img_filename_list)
    if debug:
        debug_put_lines_on_image(lines, gray_blur.copy())
        debug_put_points_on_image(points, gray_blur.copy())
        
        for x in classified:
            print(x)
    return classified


def test_detection():
    path = Path(__file__).parent.parent.parent / f"datasets/go_imgs/img/go_board_59.jpg"
    print(path)
    process_image(path)


if __name__ == '__main__':
    test_detection()