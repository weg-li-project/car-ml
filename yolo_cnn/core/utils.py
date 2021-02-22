import numpy as np
import pandas as pd

from util.paths import car_brands_filepath, car_colors_filepath


def load_weights(model, weights_file):
    layer_size = 110
    output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

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


def format_boxes(bboxes, image_height, image_width):
    '''
    Converts bounding boxes from normalized ymin, xmin, ymax, xmax to xmin, ymin, xmax, ymax in image format
    @param bboxes: bounding boxes
    @param image_height: height of the image
    @param image_width: width of the image
    @return: reformatted bounding boxes
    '''
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def get_dict_alpr():
    '''
    Get the dictionary for encoding the license plate letters.
    @return: the dictionary
    '''
    keys = get_keys_alpr()
    dict = {keys[i]: i for i in range(0, len(keys))}
    return dict


def get_keys_alpr():
    '''
    Get the keys for the encoded license plate letters.
    @return: the keys
    '''
    keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ö', 'Ü']
    return keys


def get_dict_car_brand():
    '''
    Get the dictionary for encoding the car brands.
    @return: the dictionary
    '''
    car_brands = pd.read_csv(car_brands_filepath, delimiter=',', header=None)
    dict = {car_brands.values[i, 0]: i for i in range(0, len(car_brands.values))}
    return dict


def get_keys_car_brand():
    '''
    Get the keys for the encoded car brands.
    @return: the keys
    '''
    car_brands = pd.read_csv(car_brands_filepath, delimiter=',', header=None)
    keys = list(car_brands.values[:, 0])
    return keys


def get_dict_car_color():
    '''
    Get the dictionary for encoding the car colors.
    @return: the dictionary
    '''
    car_brands = pd.read_csv(car_colors_filepath, delimiter=',', header=None)
    values = [value[0] for value in car_brands.values]
    dict = {values[i]: i for i in range(0, len(values))}
    return dict


def get_keys_car_color():
    '''
    Get the keys for the encoded car colors.
    @return: the keys
    '''
    car_brands = pd.read_csv(car_colors_filepath, delimiter=',', header=None)
    keys = list(car_brands.values[:, 0])
    return keys


def read_class_names(class_file_name):
    '''
    Read the class names from a file.
    @param class_file_name: path to the file containing the class names
    @return: class names
    '''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def containsLP(car_coords, lp_boxes):
    '''
    Test if car bounding box contains any license plate.
    @param car_coords: coordinates of car bounding box
    @param lp_boxes: license plate bounding boxes
    @return: True or False whether or not car bounding box contains any license plate
    '''
    out_boxes, out_scores, out_classes, num_boxes = lp_boxes
    for lp_coords in out_boxes:
        xmin_car, ymin_car, xmax_car, ymax_car = car_coords[0], car_coords[1], car_coords[2], car_coords[3]
        xmin_lp, ymin_lp, xmax_lp, ymax_lp = lp_coords[0], lp_coords[1], lp_coords[2], lp_coords[3]

        if xmin_car <= xmin_lp and xmax_car >= xmax_lp and ymin_car <= ymin_lp and ymax_car >= ymax_lp:
            return True

    return False
