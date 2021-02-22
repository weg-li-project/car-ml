import os
from typing import Final, List

from flask import Flask, Request, request
from werkzeug.exceptions import (BadRequest, MethodNotAllowed,
                                 UnprocessableEntity, UnsupportedMediaType)

from alpr_gcloud_vision.core.annotations import get_images_from_gcs_uris
from util.adapter import detect_car_attributes
from util.paths import checkpoints_path
from util.transforms import to_json_suggestions
from yolo_cnn.load import load_models

if os.path.exists(checkpoints_path):
    LOADED_MODELS: Final = load_models()
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

    image_data = get_images_from_gcs_uris(google_cloud_urls)
    lpns, makes, colors = detect_car_attributes(image_data)
    return to_json_suggestions(
        license_plate_numbers=lpns,
        makes=makes,
        colors=colors
    )


app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    return get_image_analysis_suggestions(request)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
