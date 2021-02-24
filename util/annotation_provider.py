import os
from typing import Union

from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse


class AnnotationProvider:
    """Client for Google Cloud Vision API."""

    def __init__(self, vision_client):
        self.vision_client = vision_client

    def get_annotations(self, image: Union[bytes, str]):
        """Return the localized object and text annotations from the Google Cloud Vision API.

        Args:
            image (Union[bytes, str]): Google Cloud Storage uri or image in memory.
        Returns:
            Localized object and text annotations.
        """

        features = [
            {"type_": vision.Feature.Type.OBJECT_LOCALIZATION},
            {"type_": vision.Feature.Type.TEXT_DETECTION},
        ]
        image = self._create_vision_image(image)

        response: AnnotateImageResponse = self.vision_client.annotate_image(
            {"image": image, "features": features}
        )

        return response.localized_object_annotations, response.text_annotations

    @staticmethod
    def _create_vision_image(image: Union[bytes, str]) -> vision.Image:
        """Check the correct type of the provided image.

        Args:
            image (Union[bytes, str]): Google Cloud Storage uri or image in memory.
        Returns:
            Image payload for the Cloud Vision API client.
        """

        if isinstance(image, bytes):
            image = vision.Image(content=image)
        elif isinstance(image, str) and image.startswith("gs://"):
            image = vision.Image(source=vision.ImageSource(gcs_image_uri=image))
        else:
            raise Exception(
                f"{__file__.split(os.path.sep)[-1]}:_create_vision_image() - Incorrect data type."
            )
        return image
