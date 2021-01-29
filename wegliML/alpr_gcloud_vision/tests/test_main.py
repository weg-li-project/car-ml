from typing import List, Dict
import pytest

from unittest.mock import Mock, patch

from werkzeug.exceptions import HTTPException

from main import get_image_analysis_suggestions, get_license_plate_number_suggestions


def request_helper(data=None, method: str = 'POST', headers: Dict[str, str] = None):
    if not headers:
        headers = {'content-type': 'application/json'}
    return Mock(get_json=Mock(return_value=data), args=data, method=method, headers=headers)


def urls_helper(urls: List[str] = None) -> Dict[str, List[str]]:
    return {'google_cloud_urls': urls}


class TestMain:
    @patch('main.get_license_plate_number_suggestions', return_value=[])
    @patch('main.to_json_suggestions', return_value='{}')
    def test_endpoint(self, mock_to_json: Mock, mock_alpr: Mock):
        data = urls_helper(['gs://weg-li_images/9876fgh'])

        response = get_image_analysis_suggestions(request_helper(data))

        mock_alpr.assert_called_once()
        mock_to_json.assert_called_once()
        assert response == '{}'

    def test_wrong_content_type(self):
        with pytest.raises(HTTPException, match='415'):
            data = urls_helper(['gs://weg-li_images/9876fgh'])
            req = request_helper(data, headers={'content-type': 'application/xml'})
            get_image_analysis_suggestions(req)

    def test_no_urls1(self):
        with pytest.raises(HTTPException, match='422'):
            data = urls_helper([])
            get_image_analysis_suggestions(request_helper(data))

    def test_no_urls2(self):
        with pytest.raises(HTTPException, match='422'):
            data = urls_helper()
            get_image_analysis_suggestions(request_helper(data))

    def test_missing_property(self):
        with pytest.raises(HTTPException, match='400'):
            data = {'"property": "value",': ',[]'}
            get_image_analysis_suggestions(request_helper(data))

    def test_malformed_json_body(self):
        with pytest.raises(HTTPException, match='400'):
            data = '¯\\_(ツ)_/¯'
            get_image_analysis_suggestions(request_helper(data))

    def test_wrong_method(self):
        with pytest.raises(HTTPException, match='405'):
            get_image_analysis_suggestions(request_helper(method='GET'))

    @patch('main.recognize_license_plate_numbers', return_value=[''])
    @patch('main.get_images_from_gcs_uris', return_value=[('', '', '')])
    def test_return_unique_license_plates(self, mock_alpr, mock_annotations):
        suggestions = get_license_plate_number_suggestions([])

        mock_alpr.assert_called_once()
        mock_annotations.assert_called_once()
        assert len(suggestions) == 1
