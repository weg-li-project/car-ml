import os

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from util.paths import (
    cnn_alpr_model_path,
    cnn_car_rec_model_path,
    cnn_color_rec_model_path,
    yolo_car_model_path,
    yolo_lp_model_path,
)
from yolo_cnn.core.cnn import CNN as CNN_alpr
from yolo_cnn.core.cnn_resnet152 import resnet152_model

# Comment out lines to enable tensorflow outputs
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "3"

yolo_lp = None
yolo_car = None
cnn_alpr = None
cnn_color_rec = None
cnn_car_rec = None


def load_models(
    yolo_lp_model_path: str = yolo_lp_model_path,
    yolo_car_model_path: str = yolo_car_model_path,
    cnn_alpr_checkpoint_path: str = cnn_alpr_model_path,
    cnn_color_rec_checkpoint_path: str = cnn_color_rec_model_path,
    cnn_car_rec_checkpoint_path: str = cnn_car_rec_model_path,
):
    """Load models and preserve them in memory for subsequent calls.
    @param yolo_lp_model_path: path to yolo model trained for license plate detection
    @param yolo_car_model_path: path to yolo model trained for car detection
    @param cnn_alpr_checkpoint_path: path to cnn model trained for letter recognition on license plates
    @param cnn_color_rec_checkpoint_path: path to resnet152 model trained for car color recognition
    @param cnn_car_rec_checkpoint_path: path to resnet152 model model trained for car brand recognition
    @return: tuple containing the yolo model trained for license plate detection, the yolo model trained for car detection, the cnn model trained for letter recognition on license plates, the resnet152 model trained for car color recognition, and the resnet152 model trained for car brand recognition
    """

    cnn_alpr_checkpoint_path: str = tf.train.latest_checkpoint(
        os.path.dirname(cnn_alpr_model_path)
    )
    cnn_color_rec_checkpoint_path: str = tf.train.latest_checkpoint(
        os.path.dirname(cnn_color_rec_model_path)
    )
    cnn_car_rec_checkpoint_path: str = tf.train.latest_checkpoint(
        os.path.dirname(cnn_car_rec_model_path)
    )

    # Load yolo lp model
    global yolo_lp
    if not yolo_lp:
        yolo_lp = tf.saved_model.load(yolo_lp_model_path, tags=[tag_constants.SERVING])

    # Load yolo car model
    global yolo_car
    if not yolo_car:
        yolo_car = tf.saved_model.load(
            yolo_car_model_path, tags=[tag_constants.SERVING]
        )

    # Load cnn alpr model
    global cnn_alpr
    if not cnn_alpr:
        cnn_alpr = CNN_alpr()
        cnn_alpr.create_model()
        cnn_alpr.load_weights(cnn_alpr_checkpoint_path)

    # Load cnn color rec model
    global cnn_color_rec
    if not cnn_color_rec:
        cnn_color_rec = resnet152_model(
            img_height=224, img_width=224, color_type=3, num_classes=10, new_model=False
        )
        cnn_color_rec.load_weights(cnn_color_rec_checkpoint_path)

    # Load cnn car rec model
    global cnn_car_rec
    if not cnn_car_rec:
        cnn_car_rec = resnet152_model(
            img_height=224, img_width=224, color_type=3, num_classes=70, new_model=False
        )
        cnn_car_rec.load_weights(cnn_car_rec_checkpoint_path)

    return yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec
