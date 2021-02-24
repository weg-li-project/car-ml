import os
from typing import Final, List

from google.cloud import storage

BUCKET_NAME: Final = os.environ["WEGLI_IMAGES_BUCKET_NAME"]


class CloudStorageClient:
    """Client for Google Cloud Storage API.
    """

    def __init__(self, storage_client=storage.Client(), bucket_name=BUCKET_NAME):
        self.storage_client = storage_client
        self.bucket_name = bucket_name

    def download_images(self, gs_uris: List[str]) -> List[bytes]:
        """Download images from Google Cloud Storage and return as list of bytes.

        Args:
            gs_uris (List[str]): Google Cloud Storage uris.
        Returns:
            Images as list of bytes.
        """

        return list(map(lambda x: self.download_image(x), gs_uris))

    def download_image(self, gs_uri: str) -> bytes:
        """Download image from Google Cloud Storage and return as bytes.

        Args:
            gs_uri (str): Google Cloud Storage uri.
        Returns:
            Image as bytes.
        """

        blob: storage.blob.Blob = self.storage_client.bucket(self.bucket_name).get_blob(
            gs_uri.replace(f"gs://{self.bucket_name}/", "")
        )
        image_in_bytes = blob.download_as_bytes()

        return image_in_bytes
