import os
from typing import Final, List

from flask import Flask, Request, request
from werkzeug.exceptions import (BadRequest, MethodNotAllowed,
                                 UnprocessableEntity, UnsupportedMediaType)

from alpr_gcloud_vision.core.annotations import get_annotations_from_gcs_uris
from util.alpr_adapter import recognize_license_plate_numbers
from util.transforms import to_json_suggestions

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


def get_license_plate_number_suggestions(google_cloud_urls: List[str]):
    annotation_data = get_annotations_from_gcs_uris(google_cloud_urls)
    return recognize_license_plate_numbers(annotation_data)


app = Flask(__name__)


@app.route("/")
def index():
    return get_image_analysis_suggestions(request)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
