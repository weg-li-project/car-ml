from google.cloud import vision
from PIL import Image

from alpr_gcloud_vision.alpr.object_detection import DetectedObject
from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate


def text_annotation(img_path):
    client = vision.ImageAnnotatorClient()

    with open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts


def recognize_license_plate(img_path, objects, texts):
    '''
    Recognize the license plate number based on the objects and texts detected by the google vision api.
    @param img_path: image path
    @param objects: objects detected by google vision api
    @param texts: texts detected by google vision api
    @return:
    '''

    cars = []
    license_plates = []

    # read image
    img = Image.open(img_path)

    for object_ in objects:
        # instantiate DetectedObject object and check whether it is of type Car, Van, Truck or License plate
        detected_object = DetectedObject(object_, img)
        if detected_object.isObject('Car') or detected_object.isObject('Van') or detected_object.isObject('Truck'):
            cars.append(detected_object)
        if detected_object.isObject('License plate'):
            license_plates.append(detected_object)

    license_plate_nos = []

    # check if license plate objects were detected by google vision api
    if len(license_plates) > 0:
        for license_plate in license_plates:

            # iterate over texts and find the bounding polygons that are contained by license plate
            license_plate_text = license_plate.findTexts(texts[1:])

            # check if detected license plate text is a valid license plate number and if so return correctly formatted license plate number
            lpc = LicensePlateCandidate(license_plate_text)
            license_plate_no, res, msg = lpc.checkCandidate()

            if res:
                license_plate_nos.append(license_plate_no)

    # if no text was found
    if len(texts) == 0:
        return []

    # evaluate the whole text and check for license plate candidates
    text_list = texts[0].description.split('\n')

    for text in text_list:
        # check if detected license plate text is a valid license plate number and if so return correctly formatted license plate number
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        # add license plate number if not already contained
        if res and not license_plate_no in license_plate_nos:
            license_plate_nos.append(license_plate_no)

    if len(license_plate_nos) > 0:
        return license_plate_nos
    else:
        return []
