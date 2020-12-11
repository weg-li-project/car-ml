from typing import List, Final

from flask import Request
from werkzeug.exceptions import UnprocessableEntity, BadRequest, MethodNotAllowed, UnsupportedMediaType

from wegliML.core.transforms import to_json_suggestions
from wegliML.core.annotations import get_annotations_from_gcs_uris
from wegliML.alpr.license_plate_recognition import recognize_license_plate

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
    license_plate_numbers = []
    for image, object_annotations, text_annotations in annotation_data:
        for license_plate_number in recognize_license_plate(image, object_annotations, text_annotations):
            license_plate_numbers.append(license_plate_number)
    return license_plate_numbers
