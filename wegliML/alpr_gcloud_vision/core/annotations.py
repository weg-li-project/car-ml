import os
from typing import List, Final, Tuple

from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from google.cloud import storage

BUCKET_NAME: Final = os.environ["WEGLI_IMAGES_BUCKET_NAME"]


def get_image_from_gcs_uri(uri: str) -> bytes:
    storage_client = storage.Client()

    blob: storage.blob.Blob = storage_client.bucket(BUCKET_NAME) \
        .get_blob(uri.replace(f'gs://{BUCKET_NAME}/', ''))
    image_in_bytes = blob.download_as_bytes()

    return image_in_bytes


def get_annotations_from_gcs_uri(image: bytes):
    client = vision.ImageAnnotatorClient()

    features = [
        {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
        {'type_': vision.Feature.Type.TEXT_DETECTION}
    ]
    image = vision.Image(content=image)

    response: AnnotateImageResponse = client.annotate_image({
        'image': image,
        'features': features
    })

    return response.localized_object_annotations, response.text_annotations


def get_images_from_gcs_uris(uris: List[str]) -> List[Tuple[str, bytes]]:
    images = []
    for uri in uris:
        images.append((uri, get_image_from_gcs_uri(uri)))
    return images
