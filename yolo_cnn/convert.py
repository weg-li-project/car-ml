
import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from core.yolo import YOLO
import core.utils as utils

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4', 'path to output')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
  batch_size = tf.shape(conv_output)[0]
  conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

  conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

  xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
  xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
  xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

  xy_grid = tf.cast(xy_grid, tf.float32)

  pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
  pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
  pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

  pred_conf = tf.sigmoid(conv_raw_conf)
  pred_prob = tf.sigmoid(conv_raw_prob)

  pred_prob = pred_conf * pred_prob
  pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
  pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

  return pred_xywh, pred_prob

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
  scores_max = tf.math.reduce_max(scores, axis=-1)

  mask = scores_max >= score_threshold
  class_boxes = tf.boolean_mask(box_xywh, mask)
  pred_conf = tf.boolean_mask(scores, mask)
  class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
  pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

  box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

  input_shape = tf.cast(input_shape, dtype=tf.float32)

  box_yx = box_xy[..., ::-1]
  box_hw = box_wh[..., ::-1]

  box_mins = (box_yx - (box_hw / 2.)) / input_shape
  box_maxes = (box_yx + (box_hw / 2.)) / input_shape
  boxes = tf.concat([
    box_mins[..., 0:1],  # y_min
    box_mins[..., 1:2],  # x_min
    box_maxes[..., 0:1],  # y_max
    box_maxes[..., 1:2]  # x_max
  ], axis=-1)
  return (boxes, pred_conf)

def save_tf():
  STRIDES = np.array([8, 16, 32])
  ANCHORS = np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]).reshape((3, 3, 2))
  XYSCALE = [1.2, 1.1, 1.05]
  NUM_CLASS = len(['license_plate'])

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS)
  bbox_tensors = []
  prob_tensors = []
  for i, fm in enumerate(feature_maps):
    if i == 0:
      output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
    elif i == 1:
      output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
    else:
      output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
    bbox_tensors.append(output_tensors[0])
    prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
  pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, FLAGS.weights)
  model.summary()
  model.save(FLAGS.output)

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
