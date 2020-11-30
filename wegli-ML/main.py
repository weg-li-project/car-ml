from typing import List, Final

from flask import Request
from werkzeug.exceptions import UnprocessableEntity, BadRequest, MethodNotAllowed, UnsupportedMediaType

from core.transform import to_json_suggestions
from license_plate_recognition import recognize_license_plate

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

    return to_json_suggestions(recognize_license_plate(google_cloud_urls))
