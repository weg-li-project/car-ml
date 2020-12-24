
import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from core.license_plate_recognizer import recognize_plate

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

def get_dict():
    keys = get_keys()
    dict = {keys[i] : i for i in range(0, len(keys))}
    return dict

def get_keys():
    keys = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Ö','Ü']
    return keys

def load_data(img_dir):
    imgs = []
    labels = []
    dict = get_dict()
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

def analyze_box(image, bboxes, model, info = False, show_label=True):
    classes = ['license_plate']
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes

    for i in range(num_boxes):

        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]

        height_ratio = int(image_h / 25)
        plate_number = recognize_plate(image, coor, model)
        if plate_number != None:
            cv2.putText(image, plate_number, (int(coor[0]), int(coor[1]-height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,0), 2)

        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if info:
            print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))

        if show_label:
            bbox_mess = '%s: %.2f' % (class_name, score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image