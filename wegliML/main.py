from typing import List, Final

import numpy as np
from flask import Request
from werkzeug.exceptions import UnprocessableEntity, BadRequest, MethodNotAllowed, UnsupportedMediaType

from alpr_gcloud_vision.core.transforms import to_json_suggestions
from alpr_gcloud_vision.core.annotations import get_annotations_from_gcs_uris
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from alpr_yolo_cnn.detect import main as alpr_yolo_cnn_main
from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate

PROP_NAME: Final = 'google_cloud_urls'

def get_image_analysis_suggestions(request: Request):
    """Responds to POST HTTP request containing a JSON body with Google Cloud Storage URLs.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        A JSON string that contains suggestions for the license plate and vehicle features.
    """
    if request.method != 'POST':
        raise MethodNotAllowed()

    content_type = request.headers['content-type']
    if content_type != 'application/json':
        raise UnsupportedMediaType()

    request_json = request.get_json()
    if request_json and PROP_NAME not in request_json:
        raise BadRequest(f"The JSON payload is invalid or is missing a '{PROP_NAME}' property")
    google_cloud_urls: List[str] = request_json[PROP_NAME]

    if not google_cloud_urls or len(google_cloud_urls) < 1:
        raise UnprocessableEntity()

    return to_json_suggestions(license_plate_numbers=get_license_plate_number_suggestions(google_cloud_urls))


def get_license_plate_number_suggestions(google_cloud_urls):
    annotation_data = get_annotations_from_gcs_uris(google_cloud_urls)
    images = [data[0] for data in annotation_data]
    plate_numbers_dict = alpr_yolo_cnn_main(images, cnn_advanced=False, yolo_checkpoint='./alpr_gcloud_vision/checkpoints/yolov4', cnn_checkpoint='./alpr_yolo_cnn/checkpoints/cnn/training')

    # check if any found license plate number is a valid license plate
    for key in plate_numbers_dict.keys():
        results = []
        for lp in plate_numbers_dict[key]:
            lpc = LicensePlateCandidate(lp)
            license_plate_no, res, msg = lpc.checkCandidate()
            results.append(res)

    if plate_numbers_dict[key] is None or plate_numbers_dict[key] == [] or not (np.array(results)).any():
        img_path = key
        for image, object_annotations, text_annotations in annotation_data:
            if image == img_path:
                license_plate_nos = recognize_license_plate(image, object_annotations, text_annotations)
                plate_numbers_dict[key] = license_plate_nos

    # remove duplicates
    license_plate_numbers_set = {}
    for l in plate_numbers_dict.values():
        license_plate_numbers_set = license_plate_numbers_set.union(set(l))
    return license_plate_numbers_set
