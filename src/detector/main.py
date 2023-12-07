from board import *
from model import model
import json

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
    if not debug:
        debug_put_lines_on_image(lines, gray_blur.copy())
        debug_put_points_on_image(points, gray_blur.copy())
        
        for x in classified:
            print(x)
    with open("./classified.json", 'w', encoding='utf-8') as f:
        json.dump({"board": classified}, f)


def test_detection():
    path = Path(__file__).parent.parent.parent / f"datasets/go_imgs/img/realimage12.jpg"
    print(path)
    process_image(path)


if __name__ == '__main__':
    if not debug:
        camera_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink"
        camera = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
        ret, image = camera.read()
        if ret:
            p = "./image_to_process.jpeg"
            cv2.imwrite(p, image)
            camera.release()
            # gc.collect()
            process_image(p)
        else:
            print("Error capturing image")
    else:
        test_detection()