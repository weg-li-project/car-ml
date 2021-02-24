from unittest.mock import Mock

from google.cloud import vision
import pytest

from tests.skip_markers import needs_google_credentials, needs_private_testdata

from util.annotation_provider import AnnotationProvider
from util.cloud_storage_client import CloudStorageClient
from util.paths import charges_schroeder_path

visionClientMock = Mock(
    annotate_image=Mock(
        return_value=Mock(localized_object_annotations=[], text_annotations=[])
    )
)
mock_image_bytes = bytes(0)
mock_image_str = "gs://bucket_name/directory/filename.extension"
storageClientMock = Mock(
    bucket=Mock(
        return_value=Mock(
            get_blob=Mock(
                return_value=Mock(download_as_bytes=Mock(return_value=mock_image_bytes))
            )
        )
    )
)


class TestAnnotations:
    def test_get_annotations(self):
        annotation_provider = AnnotationProvider(vision_client=visionClientMock)
        (
            localized_object_annotations,
            text_annotations,
        ) = annotation_provider.get_annotations(mock_image_bytes)
        assert localized_object_annotations == []
        assert text_annotations == []

    def test_get_image_from_gcs_uri(self):
        cs_client = CloudStorageClient(storage_client=storageClientMock)
        result = cs_client.download_image("")
        assert result == mock_image_bytes

    def test_create_vision_image_invalid_type(self):
        annotation_provider = AnnotationProvider(vision_client=visionClientMock)
        with pytest.raises(Exception, match="Incorrect data type."):
            annotation_provider._create_vision_image(0)

    def test_create_vision_image_gs_uri(self):
        annotation_provider = AnnotationProvider(vision_client=visionClientMock)
        result = annotation_provider._create_vision_image(mock_image_str)
        assert result == vision.Image(
            source=vision.ImageSource(gcs_image_uri=mock_image_str)
        )

    def test_create_vision_image_no_gs_uri(self):
        annotation_provider = AnnotationProvider(vision_client=visionClientMock)
        gcs_uri = "bucket_name/directory/filename.extension"
        with pytest.raises(Exception, match="Incorrect data type."):
            annotation_provider._create_vision_image(gcs_uri)

    def test_create_vision_image_bytes(self):
        annotation_provider = AnnotationProvider(vision_client=visionClientMock)
        result = annotation_provider._create_vision_image(mock_image_bytes)
        assert result == vision.Image(content=mock_image_bytes)

    @needs_google_credentials
    def test_get_annotations_gs_uri(self):
        annotation_provider = AnnotationProvider()
        gs_uri = "gs://weg-li_images/test/IMG_20191129_085112_size_big_car_one.jpg"
        (r1, r2) = annotation_provider.get_annotations(gs_uri)
        assert len(r1) > 0
        assert len(r2) > 0

    @needs_google_credentials
    @needs_private_testdata
    def test_get_annotations_bytes(self):
        filepath = charges_schroeder_path + "IMG_20191129_085112.jpg"
        annotation_provider = AnnotationProvider()
        with open(filepath, "rb") as file:
            (r1, r2) = annotation_provider.get_annotations(bytes(file.read()))
            assert len(r1) > 0
            assert len(r2) > 0
