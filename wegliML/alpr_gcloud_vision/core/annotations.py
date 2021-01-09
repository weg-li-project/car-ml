import io
import os
from typing import List, Final

from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from google.cloud import storage

BUCKET_NAME: Final = os.environ["WEGLI_IMAGES_BUCKET_NAME"]


def get_image_from_gcs_uri(uri: str):
    storage_client = storage.Client()

    blob: storage.blob.Blob = storage_client.bucket(BUCKET_NAME) \
        .get_blob(uri.replace(f'gs://{BUCKET_NAME}/', ''))
    image_in_bytes = blob.download_as_bytes()

    return image_in_bytes


def get_annotations_from_gcs_uri(uri: str):
    client = vision.ImageAnnotatorClient()
    image_in_bytes = get_image_from_gcs_uri(uri)

    features = [
        {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
        {'type_': vision.Feature.Type.TEXT_DETECTION}
    ]
    image = vision.Image(content=image_in_bytes)

    response: AnnotateImageResponse = client.annotate_image({
        'image': image,
        'features': features
    })

    return uri, io.BytesIO(image_in_bytes), response.localized_object_annotations, response.text_annotations


def get_annotations_from_gcs_uris(uris: List[str]):
    annotation_data = []
    for uri in uris:
        annotation_data.append(get_annotations_from_gcs_uri(uri))
    return annotation_data


def get_images_from_gcs_uris(uris: List[str]):
    images = []
    for uri in uris:
        images.append((uri, io.BytesIO(get_image_from_gcs_uri(uri))))
    return images
