from typing import List, Dict
import pytest

from unittest.mock import Mock, patch

from werkzeug.exceptions import HTTPException

from main import get_suggestions


storage_client_mock = Mock(download_images=Mock(return_value=[]))

detector_mock = Mock(detect_car_attributes=Mock(return_value=([], [], [])))


def request_helper(data=None, method: str = "POST", headers: Dict[str, str] = None):
    if not headers:
        headers = {"content-type": "application/json"}
    return Mock(
        get_json=Mock(return_value=data), args=data, method=method, headers=headers
    )


def urls_helper(urls: List[str] = None) -> Dict[str, List[str]]:
    return {"google_cloud_urls": urls}


class TestMain:
    @patch("main.to_json_suggestions", return_value="{}")
    def test_endpoint(self, mock_to_json: Mock):
        data = urls_helper(["gs://weg-li_images/9876fgh"])

        response = get_suggestions(
            request_helper(data),
            storage_client=storage_client_mock,
            detector=detector_mock,
        )

        detector_mock.detect_car_attributes.assert_called_once()
        mock_to_json.assert_called_once()
        storage_client_mock.download_images.assert_called_once()
        assert response == "{}"

    def test_wrong_content_type(self):
        with pytest.raises(HTTPException, match="415"):
            data = urls_helper(["gs://weg-li_images/9876fgh"])
            req = request_helper(data, headers={"content-type": "application/xml"})
            get_suggestions(
                req, storage_client=storage_client_mock, detector=detector_mock
            )

    def test_no_urls1(self):
        with pytest.raises(HTTPException, match="422"):
            data = urls_helper([])
            get_suggestions(
                request_helper(data),
                storage_client=storage_client_mock,
                detector=detector_mock,
            )

    def test_no_urls2(self):
        with pytest.raises(HTTPException, match="422"):
            data = urls_helper()
            get_suggestions(
                request_helper(data),
                storage_client=storage_client_mock,
                detector=detector_mock,
            )

    def test_missing_property(self):
        with pytest.raises(HTTPException, match="400"):
            data = {'"property": "value",': ",[]"}
            get_suggestions(
                request_helper(data),
                storage_client=storage_client_mock,
                detector=detector_mock,
            )

    def test_malformed_json_body(self):
        with pytest.raises(HTTPException, match="400"):
            data = "¯\\_(ツ)_/¯"
            get_suggestions(
                request_helper(data),
                storage_client=storage_client_mock,
                detector=detector_mock,
            )

    def test_wrong_method(self):
        with pytest.raises(HTTPException, match="405"):
            get_suggestions(
                request_helper(method="GET"),
                storage_client=storage_client_mock,
                detector=detector_mock,
            )
