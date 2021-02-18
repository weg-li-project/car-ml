
import numpy as np
import tensorflow as tf
import cv2
import random
import colorsys

from yolo_cnn.core.utils import get_keys_alpr, get_keys_car_brand, get_keys_car_color, read_class_names, containsLP
from util.paths import class_names_yolo_car

def _extract_contours(gray, thresh):
    letter_rects = []
    # find contours of regions of interest within license plate
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create copy of gray license plate
    im2 = gray.copy()
    height, width = im2.shape
    # loop through contours and find individual letters and numbers in license plate
    keep_cntrs = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
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
        return []

    prototype = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[3], reverse=True)[2]

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        _, _, prototype_width, prototype_height = cv2.boundingRect(prototype)
        if h >= prototype_height * 1.2:
            continue
        if h <= prototype_height * 0.8:
            continue
        keep_cntrs.append(cntr)

    contours = keep_cntrs

    # sort contours left-to-right
    contours = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[0])  # TODO: sort from upper left to lower right instead

    return contours

def _extract_whitespaces(contours):
    # array to store, before which rects are the 2 biggest distances
    whitespaces = [-1, -1]

    distances = []

    for i, cntr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cntr)
        if i == 0:
            dist = -float('inf')
            urc = x + w
        else:
            dist = x - urc
            urc = x + w
            distances.append((dist, i))

    distances_sorted = sorted(distances, key=lambda t: t[0], reverse=True)

    if len(distances_sorted) >= 2:
        whitespaces = [distances_sorted[0][1], distances_sorted[1][1]]

    return whitespaces

def _add_whitespaces(contours, license_plate):
    whitespaces = _extract_whitespaces(contours)
    digit_count = 0
    license_plate_whitespaces = ''
    for i in range(len(contours)):
        key = license_plate[i]
        if key.isdigit():
            if digit_count == 0:
                if i == whitespaces[0]:
                    # max distance is before 1st number -> add space before letter with 2nd largest distance, if it is before the 1st number
                    if whitespaces[1] < i:
                        license_plate_whitespaces = license_plate_whitespaces[0:whitespaces[1]] + ' ' + license_plate_whitespaces[whitespaces[1]:]
                else:
                    # max distance is not before 1st number -> add space before letter with largest distance, if it is before the 1st number
                    if whitespaces[0] < i:
                        license_plate_whitespaces = license_plate_whitespaces[0:whitespaces[0]] + ' ' + license_plate_whitespaces[whitespaces[0]:]
                license_plate_whitespaces += ' '
                digit_count += 1
        license_plate_whitespaces += key
    return license_plate_whitespaces

def _warp_license_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_wo_blue = img.copy()
    # remove blue to find contour of inner license plate (the white field) better
    img_wo_blue[:,:,0] = 0
    gray_wo_blue = cv2.cvtColor(img_wo_blue, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_wo_blue, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    # loop through contours to find license blade: should be biggest contour with w >> h etc.
    max_area = 0
    max_cntr = None
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if h == height and w == width:
            # contour is around the whole image -> skip
            continue
        if h < 20:
            # h too small -> skip
            continue
        if h * 2 > w:
            # h too large in relation to w -> skip
            continue
        if w * 2 < width:
            # w too small in relation to img -> skip
            continue
        if h * 2 < height:
            # h too small in relation to img -> skip
            continue
        area = cv2.contourArea(cntr)
        # store cntr with biggest area
        if area >= max_area:
            max_cntr = cntr
            max_area = area

    # check if area is found and if it is at last a third of the whole image
    if max_area > 0 and max_area * 3 > height * width:
        rect = cv2.minAreaRect(max_cntr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get width and height of the detected rotated rectangle
        wr = int(rect[1][0])
        hr = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, hr - 1], [0, 0], [wr - 1, 0], [wr - 1, hr - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(gray, M, (wr, hr))
        hw, ww = warped.shape
        if hw > ww:
            # warped is upright -> we have to skew it by 90 degree counter clockwise
            warped = warped.transpose()
            warped = cv2.flip(warped, 0)
        return warped

    else:
        # if contour with license plate is not found -> return original image, but in gray
        return gray

def _recognize_lp_box(img, coords, cnn_alpr):
    img_height, img_width = img.shape[:2]
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    min_width_tol = int(img_width * 0.007)
    max_width_tol = int(img_width * 0.014)
    box = img[max(int(ymin) - 5, 0):min(int(ymax) + 5, img_height), max(int(xmin) - min_width_tol, 0):min(int(xmax) + max_width_tol, img_width), :]
    # set blue channel to zero to ignore blue field with euro stars and D
    # box[:,:,0] = 0
    box_height, box_width = box.shape[:2]
    # check if boxes has correct aspect ratios
    if box_height <= box_width and box_height * 10 >= box_width:
        # grayscale region within bounding box
        # gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # resize image to three times as large as original for better readability
        box = cv2.resize(box, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # try to crop and warp the license plate and make it gray
        box = _warp_license_plate(box)

        # recalculate box_height and box_width of warped box
        box_height, box_width = box.shape

        # threshold the image using Otsus method to preprocess
        ret, thresh = cv2.threshold(box, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # letter_rects, contours = _extract_letters(lp_rect, thresh)
        contours = _extract_contours(box, thresh)

        letter_rects = []

        for i, cntr in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cntr)
            # grab character region of image
            roi = thresh[max(y - 5 - int(h * 0.13), 0):min(y + h + 5, box_height), max(x - 5, 0):min(x + w + 5, box_width)]  # some margin for error for umlauts
            # perform bitwise not to flip image to black text on white background
            roi = cv2.bitwise_not(roi)
            # perform another blur on character region
            roi = cv2.medianBlur(roi, 5)
            letter_rects.append(roi)
            # draw the rectangle
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        keys = get_keys_alpr()

        license_plate = ''
        keep_contours, keep_letter_rects = [], []
        digit_count, letter_count = 0, 0

        for i, letter_rect in enumerate(letter_rects):
            letter_rect = cv2.resize(letter_rect, dsize=(24,40))
            letter_rect = tf.convert_to_tensor(letter_rect, dtype=tf.float32)
            letter_rect = letter_rect / 255.0
            letter_rect = letter_rect[tf.newaxis, ..., tf.newaxis]

            predictions = cnn_alpr.predict(letter_rect)
            idx, confidence = np.argmax(predictions), np.max(predictions)

            if confidence >= 0.85:
                key = keys[idx]

                if key.isdigit():
                    if digit_count == 0:
                        if letter_count >= 2:
                            license_plate += key
                            digit_count += 1
                            keep_contours.append(contours[i])
                    else:
                        if digit_count < 4:
                            # if 5th digit found then ignore
                            license_plate += key
                            digit_count += 1
                            keep_contours.append(contours[i])
                elif key.isalpha():
                    # letter detected
                    if digit_count == 0 or key == 'E' or key == 'H':
                        # if letter after digit is found, take only E or H
                        license_plate += key
                        letter_count += 1
                        keep_contours.append(contours[i])

        contours = keep_contours

        license_plate_whitespaces = _add_whitespaces(contours, license_plate)

        return license_plate_whitespaces
    else:
        return ''

def _recognize_car_box(img, coords, cnn_car_rec, cnn_color_rec):
    img_height, img_width = img.shape[:2]
    xmin, ymin, xmax, ymax = coords
    min_width_tol = int(img_width * 0.007)
    max_width_tol = int(img_width * 0.014)
    box = img[max(int(ymin) - 5, 0):min(int(ymax) + 5, img_height), max(int(xmin) - min_width_tol, 0):min(int(xmax) + max_width_tol, img_width), :]
    box = cv2.resize(box, (224, 224))

    box_tensor = tf.convert_to_tensor(box)
    box_tensor /= 255
    box_tensor = box_tensor[tf.newaxis, :, :, :]

    predictions = cnn_color_rec.predict(box_tensor)
    idx, confidence = np.argmax(predictions), np.max(predictions)
    keys = get_keys_car_color()
    car_color = keys[idx]

    predictions = cnn_car_rec.predict(box_tensor)
    idx, confidence = np.argmax(predictions), np.max(predictions)
    keys = get_keys_car_brand()
    car_brand = keys[idx]

    return car_brand, car_color

def analyze_box(image, bboxes, cnn_alpr, cnn_car_rec, cnn_color_rec, info=False, case='CAR', lp_boxes=None):

    if case == 'CAR':
        classes = read_class_names(class_names_yolo_car)
    if case == 'PLATE':
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

    plate_numbers, car_brands, car_colors = [], [], []

    for i in range(num_boxes):

        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes:
            continue
        coords = out_boxes[i]
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]

        if info:
            print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coords[0], coords[1], coords[2], coords[3]))

        if case == 'PLATE' and class_name == 'license_plate':
            plate_number = _recognize_lp_box(image, coords, cnn_alpr)
            print('analyze box plate_number : {}'.format(plate_number))
            plate_numbers.append(plate_number)

        elif case == 'CAR' and class_name == 'car' or class_name == 'truck':

            # check if detected box is of adequate size to be the vehicle we are looking for
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
            box_width, box_height = xmax - xmin, ymax - ymin
            if image_h * 0.12 <= box_height and image_w * 0.12 <= box_width:

                # test if any license plate was found
                if lp_boxes and lp_boxes[3] != 0:
                    if not containsLP(coords, lp_boxes):
                        continue
                car_brand, car_color = _recognize_car_box(image, coords, cnn_car_rec, cnn_color_rec)
                if car_brand not in car_brands: car_brands.append(car_brand)
                if car_color not in car_colors: car_colors.append(car_color)

    if case == 'PLATE':
        return image, plate_numbers
    if case == 'CAR':
        return image, (car_brands, car_colors)
