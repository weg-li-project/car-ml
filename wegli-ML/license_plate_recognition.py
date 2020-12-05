
import os
import numpy as np
import pandas as pd
from google.cloud import vision
from PIL import Image
from matplotlib.patches import Polygon

from object_detection import DetectedObject, localize_objects
from license_plate_candidate import LicensePlateCandidate

def text_annotation(img_path):
    client = vision.ImageAnnotatorClient()

    with open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts

def remove_annotation_errors(license_plate_no):
    license_plate_no = license_plate_no.replace('&', ' ')
    license_plate_no = license_plate_no.replace('i', ' ')
    license_plate_no = license_plate_no.replace('.', ' ')
    # TODO: add more

    return license_plate_no

def recognize_license_plate(img_path, objects, texts):

    cars = []
    license_plates = []

    img = Image.open(img_path)

    for object_ in objects:
        detected_object = DetectedObject(object_, img)
        if detected_object.isObject('Car') or detected_object.isObject('Van'):
            cars.append(detected_object)
        if detected_object.isObject('License plate'):
            license_plates.append(detected_object)

    license_plate_nos = []

    if len(license_plates) > 0:
        # license plate was found in the image

        for license_plate in license_plates:

            # iterate over texts and find the bounding polygons that are contained by license plate
            license_plate_text = license_plate.findTexts(texts[1:])

            license_plate_text = remove_annotation_errors(license_plate_text)

            lpc = LicensePlateCandidate(license_plate_text, object_=license_plate)

            license_plate_no, res, msg = lpc.checkCandidate()

            if res:
                license_plate_nos.append(license_plate_no)

    # evaluate the whole text and check for license plate candidates
    text_list = texts[0].description.split('\n')

    for text in text_list:
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        if res and not license_plate_no in license_plate_nos:
            license_plate_nos.append(license_plate_no)

    if len(license_plate_nos) > 0:
        return license_plate_nos # TODO: sort after license plate size
    else:
        return []
