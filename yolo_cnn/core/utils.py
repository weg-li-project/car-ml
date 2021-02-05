
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm

def load_weights(model, weights_file):
    layer_size = 110
    output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    wf.close()

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

def get_dict_alpr():
    keys = get_keys_alpr()
    dict = {keys[i] : i for i in range(0, len(keys))}
    return dict

def get_keys_alpr():
    keys = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Ö','Ü']
    return keys

def get_dict_car_brand():
    car_brands = pd.read_csv('car_brands.txt', delimiter=',', header=None)
    dict = {car_brands.values[i, 0]: i for i in range(0, len(car_brands.values))}
    return dict

def get_keys_car_brand():
    car_brands = pd.read_csv('car_brands.txt', delimiter=',', header=None)
    keys = list(car_brands.values[:, 0])
    return keys

def get_dict_car_color():
    car_brands = pd.read_csv('car_colors.txt', delimiter=',', header=None)
    values = [value[0] for value in car_brands.values]
    dict = {values[i]: i for i in range(0, len(values))}
    return dict

def get_keys_car_color():
    car_brands = pd.read_csv('car_colors.txt', delimiter=',', header=None)
    keys = list(car_brands.values[:, 0])
    return keys

def load_letter_data(img_dir):
    imgs = []
    labels = []
    dict = get_dict_alpr()
    img_subdirs = [img_dir + '/' + 'letters_' + key for key in dict.keys()]
    for img_subdir in tqdm(img_subdirs, desc='load data'):
        for i, file in enumerate(os.listdir(img_subdir)):
            if file.endswith('.jpg'):
                    labels.append(file.replace('.jpg', '')[-1])
                    img = cv2.imread(img_subdir + '/' + file, 0)
                    img = cv2.resize(img, dsize=(24,40))
                    imgs.append(tf.convert_to_tensor(img))

    X = np.stack(imgs)
    y = np.asarray([(x in dict.keys() and dict[x]) for x in labels])

    return X, y