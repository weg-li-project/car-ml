import io
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
from PIL import Image
from tqdm import tqdm

from yolo_cnn.core.utils import format_boxes
from yolo_cnn.core.license_plate_recognizer import analyze_box
from yolo_cnn.core.cnn_alpr import CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow outputs

LOADED_YOLO_MODEL = None
LOADED_CNN_MODEL = None


def load_models(yolo_model_path: str, use_cnn_advanced: bool, latest_cnn_checkpoint_path: str):
    """Load models and preserve them in memory for subsequent calls."""

    # Load YOLO model
    global LOADED_YOLO_MODEL
    if not LOADED_YOLO_MODEL:
        LOADED_YOLO_MODEL = tf.saved_model.load(yolo_model_path, tags=[tag_constants.SERVING])

    # Load CNN model
    global LOADED_CNN_MODEL
    if not LOADED_CNN_MODEL:
        LOADED_CNN_MODEL = CNN()
        LOADED_CNN_MODEL.create_model(use_cnn_advanced)
        LOADED_CNN_MODEL.load_weights(latest_cnn_checkpoint_path)

    return LOADED_YOLO_MODEL, LOADED_CNN_MODEL


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(uris, images, output=None, show=False, cnn_advanced=False, yolo_checkpoint='./checkpoints/yolo_lp', cnn_checkpoint='./checkpoints/cnn_alpr/training'):
    # Load models
    latest = tf.train.latest_checkpoint(os.path.dirname(cnn_checkpoint))
    saved_model_loaded, model = load_models(yolo_checkpoint, cnn_advanced, latest)

    plate_numbers_dict = {}
    for index, img in enumerate(tqdm(images)):
        plate_numbers_dict.update(detect_recognize_plate(model, uris[index], img, saved_model_loaded, output, show))

    return plate_numbers_dict


def detect_recognize_plate(model, img_path, image, saved_model_loaded, output=None, show=False, info=False):
    input_size = 416

    # loop through images in list and run Yolov4 model on each
    # for count, img_path in enumerate(tqdm(images)):
    original_image = cv2.imdecode(np.fromstring(io.BytesIO(image).read(), np.uint8), cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
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

    image, recognized_plate_numbers = analyze_box(original_image, pred_bbox, model, info)

    image = Image.fromarray(image.astype(np.uint8))
    if show:
        image.show()

    plate_numbers_dict = {img_path: recognized_plate_numbers}

    # if output is not None:
    #     image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(os.path.join(output, 'detection' + str(count) + '.png'), image)

    return plate_numbers_dict
