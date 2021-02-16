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

from util.paths import yolo_lp_model_path, yolo_car_model_path, cnn_alpr_model_path, cnn_color_rec_model_path, cnn_car_rec_model_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow outputs

yolo_lp = None
yolo_car = None
cnn_alpr = None
cnn_color_rec = None
cnn_car_rec = None


def load_models(yolo_lp_model_path: str, yolo_car_model_path: str, cnn_alpr_checkpoint_path: str, cnn_color_rec_checkpoint_path: str, cnn_car_rec_checkpoint_path: str):
    """Load models and preserve them in memory for subsequent calls."""

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
        cnn_color_rec = resnet152_model(img_rows=224, img_cols=224, color_type=3, num_classes=10)
        cnn_color_rec.load_weights(cnn_color_rec_checkpoint_path)

    # Load cnn car rec model
    global cnn_car_rec
    if not cnn_car_rec:
        cnn_car_rec = resnet152_model(img_rows=224, img_cols=224, color_type=3, num_classes=70)
        cnn_car_rec.load_weights(cnn_car_rec_checkpoint_path)

    return yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def detect_object(image, yolo):
    input_size = 416

    original_image = cv2.imdecode(np.frombuffer(io.BytesIO(image).read(), np.uint8), cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = np.asarray([image_data]).astype(np.float32)

    infer = yolo.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    return original_image, pred_bbox


def detect_recognize_plate(cnn_alpr, cnn_car_rec, cnn_color_rec, img_path, image, yolo, info=False):
    original_image, pred_bbox = detect_object(image, yolo)
    image, recognized_plate_numbers = analyze_box(original_image, pred_bbox, cnn_alpr, cnn_car_rec, cnn_color_rec, info)
    plate_numbers_dict = {img_path: recognized_plate_numbers}
    return plate_numbers_dict


def detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, img_path, image, yolo, info=False):
    original_image, pred_bbox = detect_object(image, yolo)
    image, (car_brands, car_colors) = analyze_box(original_image, pred_bbox, cnn_alpr, cnn_car_rec, cnn_color_rec, info)
    return car_brands, car_colors


def main(uris, images, yolo_lp=yolo_lp_model_path, yolo_car=yolo_car_model_path, cnn_alpr=cnn_alpr_model_path, cnn_color_rec=cnn_color_rec_model_path, cnn_car_rec=cnn_car_rec_model_path):
    # Load models
    cnn_alpr = tf.train.latest_checkpoint(os.path.dirname(cnn_alpr))
    cnn_color_rec = tf.train.latest_checkpoint(os.path.dirname(cnn_color_rec))
    cnn_car_rec = tf.train.latest_checkpoint(os.path.dirname(cnn_car_rec))
    yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec = load_models(yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec)

    plate_numbers_dict = {}
    car_brands_dict = {}
    car_colors_dict = {}
    for index, img in enumerate(images):
        plate_numbers_dict.update(detect_recognize_plate(cnn_alpr, cnn_car_rec, cnn_color_rec, uris[index], img, yolo_lp))
        car_brands, car_colors = detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, uris[index], img, yolo_car)
        car_brands_dict.update({uris[index]: car_brands})
        car_colors_dict.update({uris[index]: car_colors})

    return plate_numbers_dict, car_brands_dict, car_colors_dict
