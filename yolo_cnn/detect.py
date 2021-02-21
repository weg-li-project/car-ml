import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2

from yolo_cnn.core.utils import format_boxes
from yolo_cnn.core.recognizer import analyze_box
from yolo_cnn.core.cnn import CNN as CNN_alpr
from yolo_cnn.core.cnn_resnet152 import resnet152_model

from util.paths import yolo_lp_model_path, yolo_car_model_path, cnn_alpr_model_path, cnn_color_rec_model_path, \
    cnn_car_rec_model_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # comment out below line to enable tensorflow outputs

yolo_lp = None
yolo_car = None
cnn_alpr = None
cnn_color_rec = None
cnn_car_rec = None


def load_models(yolo_lp_model_path: str, yolo_car_model_path: str, cnn_alpr_checkpoint_path: str,
                cnn_color_rec_checkpoint_path: str, cnn_car_rec_checkpoint_path: str):
    '''
    Load models and preserve them in memory for subsequent calls.
    @param yolo_lp_model_path: path to yolo model trained for license plate detection
    @param yolo_car_model_path: path to yolo model trained for car detection
    @param cnn_alpr_checkpoint_path: path to cnn model trained for letter recognition on license plates
    @param cnn_color_rec_checkpoint_path: path to resnet152 model trained for car color recognition
    @param cnn_car_rec_checkpoint_path: path to resnet152 model model trained for car brand recognition
    @return: tuple containing the yolo model trained for license plate detection, the yolo model trained for car detection, the cnn model trained for letter recognition on license plates, the resnet152 model trained for car color recognition, and the resnet152 model trained for car brand recognition
    '''

    # Load yolo lp model
    global yolo_lp
    if not yolo_lp:
        yolo_lp = tf.saved_model.load(yolo_lp_model_path, tags=[tag_constants.SERVING])

    # Load yolo car model
    global yolo_car
    if not yolo_car:
        yolo_car = tf.saved_model.load(yolo_car_model_path, tags=[tag_constants.SERVING])

    # Load cnn alpr model
    global cnn_alpr
    if not cnn_alpr:
        cnn_alpr = CNN_alpr()
        cnn_alpr.create_model()
        cnn_alpr.load_weights(cnn_alpr_checkpoint_path)

    # Load cnn color rec model
    global cnn_color_rec
    if not cnn_color_rec:
        cnn_color_rec = resnet152_model(img_height=224, img_width=224, color_type=3, num_classes=10, new_model=False)
        cnn_color_rec.load_weights(cnn_color_rec_checkpoint_path)

    # Load cnn car rec model
    global cnn_car_rec
    if not cnn_car_rec:
        cnn_car_rec = resnet152_model(img_height=224, img_width=224, color_type=3, num_classes=70, new_model=False)
        cnn_car_rec.load_weights(cnn_car_rec_checkpoint_path)

    return yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def detect_object(img, yolo, threshold):
    '''
    Detect specific object in image.
    @param img: image
    @param yolo: yolo model trained to detect specific object
    @param threshold: threshold for yolo detection score, all detections with a confidence below the threshold are rejected
    @return: tuple containing the image and the bounding boxes of the detected objects
    '''
    # load, convert and normalize image
    img = cv2.imdecode(np.frombuffer(io.BytesIO(img).read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = cv2.resize(img, (416, 416))
    img_data = img_data / 255.
    images_data = np.asarray([img_data]).astype(np.float32)

    infer = yolo.signatures['serving_default']
    batch_data = tf.constant(images_data)
    # detect objects in image using the yolo model
    pred_bboxes = infer(batch_data)
    for key, value in pred_bboxes.items():
        bboxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    bboxes, scores, classes, num_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=threshold
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax to xmin, ymin, xmax, ymax in image format
    img_h, img_w, _ = img.shape
    bboxes = format_boxes(bboxes.numpy()[0], img_h, img_w)

    # hold all detection data in one variable
    pred_bboxes = [bboxes, scores.numpy()[0], classes.numpy()[0], num_detections.numpy()[0]]

    return img, pred_bboxes


def detect_recognize_plate(cnn_alpr, cnn_car_rec, cnn_color_rec, img_path, img, yolo, info=False):
    '''
    Detect the license plates in the image and recognize the license plate numbers.
    @param cnn_alpr: cnn model trained for letter recognition on license plates
    @param cnn_car_rec: resnet152 model trained for car brand recognition
    @param cnn_color_rec: resnet152 model trained for car color recognition
    @param img_path: image paths
    @param img: image object
    @param yolo: yolo model trained for license plate detection
    @param info: whether or not some information should be displayed
    @return: dictionary containing the license plate numbers detected and recognized on the images and the bounding boxes of those license plates
    '''
    img, pred_bboxes_lp = detect_object(img, yolo, threshold=0.5)
    img, recognized_plate_numbers = analyze_box(img, pred_bboxes_lp, cnn_alpr, cnn_car_rec, cnn_color_rec, info=info,
                                                case='PLATE')
    plate_numbers_dict = {img_path: recognized_plate_numbers}
    return plate_numbers_dict, pred_bboxes_lp


def detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, img_path, img, yolo, pred_bboxes_lp, info=False):
    '''
    Detect the cars in the image and recognize the car brands and car colors.
    @param cnn_alpr: cnn model trained for letter recognition on license plates
    @param cnn_car_rec: resnet152 model trained for car brand recognition
    @param cnn_color_rec: resnet152 model trained for car color recognition
    @param img_path: image paths
    @param img: image object
    @param yolo: yolo model trained for car detection
    @param pred_bboxes_lp: bounding boxes of detected license plates in the image
    @param info: whether or not some information should be displayed
    @return: tuple of lists containing the car brands and car colors
    '''
    img, pred_bboxes_car = detect_object(img, yolo, threshold=0.5)
    img, (car_brands, car_colors) = analyze_box(img, pred_bboxes_car, cnn_alpr, cnn_car_rec, cnn_color_rec, info=info,
                                                case='CAR', pred_bboxes_lp=pred_bboxes_lp)
    return car_brands, car_colors


def main(uris, imgs, yolo_lp_model_path=yolo_lp_model_path, yolo_car_model_path=yolo_car_model_path,
         cnn_alpr_model_path=cnn_alpr_model_path, cnn_color_rec_model_path=cnn_color_rec_model_path,
         cnn_car_rec_model_path=cnn_car_rec_model_path):
    '''

    @param uris: image paths in google cloud storage
    @param imgs: images
    @param yolo_lp_model_path: path to yolo model trained for license plate detection
    @param yolo_car_model_path: path to yolo model trained for car detection
    @param cnn_alpr_checkpoint_path: path to cnn model trained for letter recognition on license plates
    @param cnn_color_rec_checkpoint_path: path to resnet152 model trained for car color recognition
    @param cnn_car_rec_checkpoint_path: path to resnet152 model model trained for car brand recognition
    @return:
    '''
    # Load models
    cnn_alpr = tf.train.latest_checkpoint(os.path.dirname(cnn_alpr_model_path))
    cnn_color_rec = tf.train.latest_checkpoint(os.path.dirname(cnn_color_rec_model_path))
    cnn_car_rec = tf.train.latest_checkpoint(os.path.dirname(cnn_car_rec_model_path))
    yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec = load_models(yolo_lp_model_path, yolo_car_model_path,
                                                                          cnn_alpr, cnn_color_rec, cnn_car_rec)

    plate_numbers_dict = {}
    car_brands_dict = {}
    car_colors_dict = {}
    for index, img in enumerate(imgs):
        plate_numbers, pred_bboxes_lp = detect_recognize_plate(cnn_alpr, cnn_car_rec, cnn_color_rec, uris[index], img,
                                                               yolo_lp)
        car_brands, car_colors = detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, uris[index], img, yolo_car,
                                                      pred_bboxes_lp)
        plate_numbers_dict.update(plate_numbers)
        car_brands_dict.update({uris[index]: car_brands})
        car_colors_dict.update({uris[index]: car_colors})

    return plate_numbers_dict, car_brands_dict, car_colors_dict
