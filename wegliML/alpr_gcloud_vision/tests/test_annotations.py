import os
from unittest.mock import Mock, patch

from alpr_gcloud_vision.core.annotations import get_image_from_gcs_uri
from alpr_gcloud_vision.core.annotations import get_annotations_from_gcs_uri

visionClientMock = Mock(
    annotate_image=Mock(
        return_value=Mock(localized_object_annotations=[], text_annotations=[])
    )
)
mock_image_bytes = bytes(0)
storageClientMock = Mock(
    bucket=Mock(
        return_value=Mock(
            get_blob=Mock(
                return_value=Mock(
                    download_as_bytes=Mock(return_value=mock_image_bytes)
                )
            )
        )
    )
)


class TestAnnotations:
    @patch('alpr_gcloud_vision.core.annotations.get_image_from_gcs_uri', return_value=bytes(0))
    @patch('alpr_gcloud_vision.core.annotations.vision.ImageAnnotatorClient', return_value=visionClientMock)
    def test_get_annotations_from_gcs_uri(self, mock1, mock2):
        localized_object_annotations, text_annotations = get_annotations_from_gcs_uri(mock_image_bytes)
        assert localized_object_annotations == []
        assert text_annotations == []

    @patch('alpr_gcloud_vision.core.annotations.storage.Client', return_value=storageClientMock)
    def test_get_image_from_gcs_uri(self, mock1):
        result = get_image_from_gcs_uri("")
        assert result == mock_image_bytes
