import os
from typing import Final, List

from flask import Flask, Request, request
from google.cloud import storage, vision
from werkzeug.exceptions import (
    BadRequest,
    MethodNotAllowed,
    UnprocessableEntity,
    UnsupportedMediaType,
)

from util.detection_adapter import DetectionAdapter
from util.cloud_storage_client import CloudStorageClient
from util.paths import checkpoints_path
from util.transforms import to_json_suggestions
from yolo_cnn.load import load_models

#if os.path.exists(checkpoints_path):
#    LOADED_MODELS: Final = load_models()
PROP_NAME: Final = "google_cloud_urls"


def get_suggestions(req: Request, storage_client, detector):
    """Responds to POST HTTP request containing a JSON body with Google Cloud Storage URLs.

    Args:
        req (flask.Request): HTTP request object.
        storage_client (CloudStorageClient): Client for handling Google Cloud Storage operations.
        detector: Container with method to detect car attributes from provided images.
    Returns:
        A JSON string that contains suggestions for the license plate and vehicle features.
    """
    if req.method != "POST":
        raise MethodNotAllowed()

    content_type = req.headers["content-type"]
    if content_type != "application/json":
        raise UnsupportedMediaType()

    request_json = req.get_json()
    if request_json and PROP_NAME not in request_json:
        raise BadRequest(
            f"The JSON payload is invalid or is missing a '{PROP_NAME}' property"
        )
    google_cloud_urls: List[str] = request_json[PROP_NAME]

    if not google_cloud_urls or len(google_cloud_urls) < 1:
        raise UnprocessableEntity()

    images = storage_client.download_images(google_cloud_urls)
    lpns, makes, colors = detector.detect_car_attributes(images, google_cloud_urls)
    return to_json_suggestions(license_plate_numbers=lpns, makes=makes, colors=colors)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    cs_client = CloudStorageClient(storage.Client())
    detector = DetectionAdapter(vision.ImageAnnotatorClient())
    return get_suggestions(request, storage_client=cs_client, detector=detector)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
