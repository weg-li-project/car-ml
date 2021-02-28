import numpy as np
import tensorflow as tf
import cv2
import random
import colorsys

from yolo_cnn.core.utils import get_keys_alpr, get_keys_car_brand, get_keys_car_color, read_class_names, containsLP
from util.paths import class_names_yolo_car


def _extract_contours(box, thresh):
    '''
    Find and extract letter contours in the box image.
    @param box: the box image
    @param thresh: thresholded version of the box image
    @return: the sorted list of contours detected
    '''
    # create copy of gray license plate image
    box_cp = box.copy()
    height, width = box_cp.shape

    # find contours of regions of interest within license plate
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop through contours and remove contours that do not match the size and width of letters
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

    # if there are less than 3 contours this is not a valid license plate number
    if len(contours) < 3:
        return []

    # sort contours by height and get prototype as third biggest cnt
    prototype = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[3], reverse=True)[2]

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        _, _, prototype_width, prototype_height = cv2.boundingRect(prototype)
        # compare contour height to prototype height
        if h >= prototype_height * 1.2:
            continue
        if h <= prototype_height * 0.8:
            continue
        keep_cntrs.append(cntr)

    contours = keep_cntrs

    # sort contours left-to-right
    contours = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[0])

    return contours


def _extract_whitespaces(contours):
    '''
    Get the positions of where to add the whitespaces in the license plate number as the two biggest distances between the contours.
    @param contours: contours of letters
    @return: the positions of the whitespaces in the license plate number
    '''
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


def _add_whitespaces(contours, license_plate_number):
    '''
    Add whitespaces to license plate number.
    @param contours: contours of letters
    @param license_plate_number: the license plate number without whitespaces
    @return:
    '''
    # get the positions of where to add the whitespaces in the license plate number
    whitespaces = _extract_whitespaces(contours)
    digit_count = 0
    license_plate_whitespaces = ''
    for i in range(len(contours)):
        key = license_plate_number[i]
        if key.isdigit():
            if digit_count == 0:
                if i == whitespaces[0]:
                    # max distance is before 1st number -> add space before letter with 2nd largest distance, if it is before the 1st number
                    if whitespaces[1] < i:
                        license_plate_whitespaces = license_plate_whitespaces[
                                                    0:whitespaces[1]] + ' ' + license_plate_whitespaces[whitespaces[1]:]
                else:
                    # max distance is not before 1st number -> add space before letter with largest distance, if it is before the 1st number
                    if whitespaces[0] < i:
                        license_plate_whitespaces = license_plate_whitespaces[
                                                    0:whitespaces[0]] + ' ' + license_plate_whitespaces[whitespaces[0]:]
                license_plate_whitespaces += ' '
                digit_count += 1
        license_plate_whitespaces += key
    return license_plate_whitespaces


def _warp_license_plate(img):
    '''
    Warp license plate image to contain an even license plate.
    @param img: image
    @return: warped image or grayscaled image if warp is not possible
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_wo_blue = img.copy()

    # remove blue to find contour of inner license plate (the white field) better
    img_wo_blue[:, :, 0] = 0
    gray_wo_blue = cv2.cvtColor(img_wo_blue, cv2.COLOR_RGB2GRAY)

    # threshold the image
    ret, thresh = cv2.threshold(gray_wo_blue, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    # find the contour on which to align the license plate
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape

    # loop through contours to find license plate: should be biggest contour with w >> h etc.
    max_area = 0
    max_cntr = None
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        # if contour is around the whole image -> skip
        if h == height and w == width:
            continue
        # if h too small -> skip
        if h < 20:
            continue
        # if h too large in relation to w -> skip
        if h * 2 > w:
            continue
        # if w too small in relation to img -> skip
        if w * 2 < width:
            continue
        # if h too small in relation to img -> skip
        if h * 2 < height:
            continue
        area = cv2.contourArea(cntr)
        # store contour with biggest area
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
    '''
    Recognize the license plate number contained in the specified bounding box in the image.
    @param img: the image
    @param coords: the coordinates of the bounding box
    @param cnn_alpr: cnn trained on letter recognition
    @return: the detected license plate number with whitespaces
    '''
    img_height, img_width = img.shape[:2]
    xmin, ymin, xmax, ymax = coords

    # get the subimage that makes up the bounded region and take some additional pixels on each side
    min_width_tol = int(img_width * 0.007)
    max_width_tol = int(img_width * 0.014)
    box = img[max(int(ymin) - 5, 0):min(int(ymax) + 5, img_height),
          max(int(xmin) - min_width_tol, 0):min(int(xmax) + max_width_tol, img_width), :]

    box_height, box_width = box.shape[:2]

    # check if box has correct aspect ratios
    if box_height <= box_width and box_height * 10 >= box_width:

        # resize image to three times as large as original for better readability
        box = cv2.resize(box, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # try to crop and warp the license plate and make it gray
        box = _warp_license_plate(box)

        # recalculate box_height and box_width of warped box
        box_height, box_width = box.shape

        # threshold the image using Otsus method to preprocess
        ret, thresh = cv2.threshold(box, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # find and extract letter contours in the box image
        contours = _extract_contours(box, thresh)

        letter_rects = []

        for i, cntr in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cntr)
            # grab character region of image (note: we leave some margin for error for Ö, Ü)
            roi = thresh[max(y - 5 - int(h * 0.13), 0):min(y + h + 5, box_height),
                  max(x - 5, 0):min(x + w + 5, box_width)]
            # perform bitwise not to flip image to black text on white background
            roi = cv2.bitwise_not(roi)
            # perform another blur on character region
            roi = cv2.medianBlur(roi, 5)
            letter_rects.append(roi)

        keys = get_keys_alpr()

        license_plate_number = ''
        keep_contours, keep_letter_rects = [], []
        digit_count, letter_count = 0, 0

        for i, letter_rect in enumerate(letter_rects):
            letter_rect = cv2.resize(letter_rect, dsize=(24, 40))
            letter_rect = tf.convert_to_tensor(letter_rect, dtype=tf.float32)
            letter_rect = letter_rect / 255.0
            letter_rect = letter_rect[tf.newaxis, ..., tf.newaxis]

            # recognize letter
            predictions = cnn_alpr.predict(letter_rect)
            idx, confidence = np.argmax(predictions), np.max(predictions)

            # put together the license plate number based on the predicted letters
            if confidence >= 0.85:
                key = keys[idx]
                if key.isdigit():
                    if digit_count == 0:
                        if letter_count >= 2:
                            license_plate_number += key
                            digit_count += 1
                            keep_contours.append(contours[i])
                    else:
                        if digit_count < 4:
                            # if 5th digit found then ignore
                            license_plate_number += key
                            digit_count += 1
                            keep_contours.append(contours[i])
                elif key.isalpha():
                    # letter detected
                    if digit_count == 0 or key == 'E' or key == 'H':
                        # if letter after digit is found, take only E or H
                        license_plate_number += key
                        letter_count += 1
                        keep_contours.append(contours[i])

        contours = keep_contours

        # add whitespaces to license plate number
        license_plate_whitespaces = _add_whitespaces(contours, license_plate_number)

        return license_plate_whitespaces
    else:
        return ''


def _recognize_car_box(img, coords, cnn_car_rec, cnn_color_rec):
    '''
    Recognize the car brand and the car color of the car contained in the specified bounding box in the image.
    @param img: the image
    @param coords: the coordinates of the bounding box
    @param cnn_car_rec: resnet152 model trained on car brand recognition
    @param cnn_color_rec: resnet152 model trained on car color recognition
    @return: tuple of the detected car brand and car color
    '''
    img_height, img_width = img.shape[:2]
    xmin, ymin, xmax, ymax = coords

    # get the subimage that makes up the bounded region and take some additional pixels on each side
    min_width_tol = int(img_width * 0.007)
    max_width_tol = int(img_width * 0.014)
    box = img[max(int(ymin) - 5, 0):min(int(ymax) + 5, img_height),
          max(int(xmin) - min_width_tol, 0):min(int(xmax) + max_width_tol, img_width), :]

    # resize, convert and normalize the image to fit the models
    box = cv2.resize(box, (224, 224))
    box_tensor = tf.convert_to_tensor(box)
    box_tensor /= 255
    box_tensor = box_tensor[tf.newaxis, :, :, :]

    # predict the car color
    predictions = cnn_color_rec.predict(box_tensor)
    idx, confidence = np.argmax(predictions), np.max(predictions)
    # decode one hot encoded tensor prediction
    keys = get_keys_car_color()
    car_color = keys[idx]

    # predict the car brand
    predictions = cnn_car_rec.predict(box_tensor)
    idx, confidence = np.argmax(predictions), np.max(predictions)
    # decode one hot encoded tensor prediction
    keys = get_keys_car_brand()
    car_brand = keys[idx]

    return car_brand, car_color


def analyze_box(image, pred_bboxes, cnn_alpr, cnn_car_rec, cnn_color_rec, info=False, case='CAR', pred_bboxes_lp=None):
    '''
    Analyze the objects detected in the bounding boxes to recognize the license plate number or car brand and car color.
    @param image: image
    @param pred_bboxes: predicted bounding boxes
    @param cnn_alpr: cnn trained on letter recognition
    @param cnn_car_rec: resnet152 model trained on car brand recognition
    @param cnn_color_rec: resnet152 model trained on car color recognition
    @param info: whether info should be displayed or not
    @param case: whether we are analyzing a license plate box or a car box
    @param pred_bboxes_lp: license plate bounding boxes
    @return: a tuple containing the image and either the license plate numbers or the car brands and car colors depending on the case
    '''
    # read the class names that the respective yolo model was trained on
    if case == 'CAR':
        classes = read_class_names(class_names_yolo_car)
    if case == 'PLATE':
        classes = ['license_plate']

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    bboxes, scores, out_classes, num_boxes = pred_bboxes
    plate_numbers, car_brands, car_colors = [], [], []
    for i in range(num_boxes):

        if int(out_classes[i]) >= 0 and int(out_classes[i]) <= num_classes:

            coords = bboxes[i]
            score = scores[i]
            class_idx = int(out_classes[i])
            class_name = classes[class_idx]

            if info:
                print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(
                    class_name, score, coords[0], coords[1], coords[2], coords[3]))

            if case == 'PLATE' and class_name == 'license_plate':
                # recognize license plate number that is contained in the bounding box
                plate_number = _recognize_lp_box(image, coords, cnn_alpr)
                plate_numbers.append(plate_number)

            elif case == 'CAR' and class_name == 'car' or class_name == 'truck' or class_name == 'bus':

                # sort out boxes of wrong size to be a vehicle
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
                box_width, box_height = xmax - xmin, ymax - ymin
                if image_h * 0.12 <= box_height and image_w * 0.12 <= box_width:

                    # test if any license plate was found and if so car bounding box contains any license plate
                    if pred_bboxes_lp and pred_bboxes_lp[3] != 0 and not containsLP(coords, pred_bboxes_lp):
                        continue
                    # recognize car brand and car color of the car contained in the bounding box
                    car_brand, car_color = _recognize_car_box(image, coords, cnn_car_rec, cnn_color_rec)
                    # add car brand and car color to lists if they are not already contained
                    if car_brand not in car_brands: car_brands.append(car_brand)
                    if car_color not in car_colors: car_colors.append(car_color)

    if case == 'PLATE':
        return image, plate_numbers
    if case == 'CAR':
        return image, (car_brands, car_colors)
