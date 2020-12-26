
import random
import numpy as np
import cv2
import colorsys
import tensorflow as tf

from core.utils import get_keys

def extract_letters(gray, thresh):
    letter_rects = []
    # find contours of regions of interest within license plate
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create copy of gray license plate
    im2 = gray.copy()
    height, width = im2.shape
    # loop through contours and find individual letters and numbers in license plate
    keep_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if h < 20:
            continue
        if h < w:
            continue
        # if height of box is not tall enough relative to total height then skip
        if h < 0.2 * height:
            continue
        # if height to width ratio is less than 25% skip
        if w < h * 0.25:
            continue
        # if width is not wide enough relative to total width then skip
        if width / w > 25:
            continue
        keep_cntrs.append(cntr)

    contours = keep_cntrs
    keep_cntrs = []

    # sort contours by height and get prototype as third biggest cnt
    if len(contours) < 3:
        return [], -1

    prototype = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[3], reverse=True)[2]

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        _, _, prototype_width, prototype_height = cv2.boundingRect(prototype)
        if h >= prototype_height * 1.1:
            continue
        if h <= prototype_height * 0.8:
            continue
        keep_cntrs.append(cntr)

    contours = keep_cntrs

    # sort contours left-to-right
    contours = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[0]) # TODO: sort from upper left to lower right instead

    whitespaces = [-1, -1]

    for i, cntr in enumerate(contours):

        x, y, w, h = cv2.boundingRect(cntr)
        if i == 0:
            dist = -float('inf')
            distances = []
            urc = x + w
        else:
            dist = x - urc
            urc = x + w

        distances.append((dist, i))

        # grab character region of image
        roi = thresh[max(y - 5 - int(h * 0.13), 0):min(y + h + 5, height), max(x - 5, 0):min(x + w + 5, width)]  # some margin for error for umlauts
        # perform bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        letter_rects.append(roi)

    distances_sorted = sorted(distances, key=lambda t: t[0], reverse=True)
    if len(distances_sorted) >= 2:
        whitespaces = [distances_sorted[0][1], distances_sorted[1][1]]
    return letter_rects, whitespaces

def recognize_plate(img, coords, model):
    img_height, img_width = img.shape[:2]
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    width_tol = int(img_width * 0.011)
    box = img[max(int(ymin) - 5, 0):min(int(ymax) + 5, img_height), max(int(xmin) - 0, 0):min(int(xmax) + width_tol, img_width), :]
    # set blue channel to zero to ignore blue field with euro stars and D
    box[:,:,0] = 0
    box_height, box_width = box.shape[:2]
    # check if boxes has correct aspect ratios
    if box_height <= box_width and box_height * 10 >= box_width:
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # perform gaussian blur to smoothen image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image using Otsus method to preprocess
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        letter_rects, whitespaces = extract_letters(gray, thresh)

        keys = get_keys()

        firstDigit = False
        license_plate = ''
        for i, letter_rect in enumerate(letter_rects):

            letter_rect = cv2.resize(letter_rect, dsize=(24,40))
            letter_rect = tf.convert_to_tensor(letter_rect, dtype=tf.float32)
            letter_rect = letter_rect / 255.0
            letter_rect = letter_rect[tf.newaxis, ..., tf.newaxis]
            key = keys[np.argmax(model.predict(letter_rect))]
            if key.isdigit() and firstDigit == False:
                if i == whitespaces[0]:
                    license_plate = license_plate[0:whitespaces[1]] + ' ' + license_plate[whitespaces[1]:]
                else:
                    license_plate = license_plate[0:whitespaces[0]] + ' ' + license_plate[whitespaces[0]:]
                license_plate += ' '
                firstDigit = True
            license_plate += key

        return license_plate

def analyze_box(image, bboxes, model, info=False, show_label=True):
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
            cv2.putText(image, plate_number, (int(coor[0]), int(coor[1] - height_ratio)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (255, 255, 0), 2)

        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if info:
            print(
                "Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(
                    class_name, score, coor[0], coor[1], coor[2], coor[3]))

        if show_label:
            bbox_mess = '%s: %.2f' % (class_name, score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image
