from unittest.mock import patch, Mock

from util.detection_adapter import DetectionAdapter

plate_no_1 = "HH MC 1210"
plate_no_2 = "PI BA 2755"
plate_number_dict = {"/0.jpg": [plate_no_1, plate_no_2], "/1.jpg": [plate_no_2]}


class TestDetectionAdapter:
    @patch(
        "util.detection_adapter.alpr_yolo_cnn_main",
        return_value=(plate_number_dict, {}, {}),
    )
    def test_detect_car_attributes(self, mock1):
        car_attributes = DetectionAdapter(Mock()).detect_car_attributes([], [])
        assert car_attributes == ([plate_no_2, plate_no_1], [], [])

    @patch("util.detection_adapter.recognize_license_plate", return_value=["LP B 542"])
    def test_cloud_vision_fallback_empty_list(self, recognize_license_plate):
        adapter = DetectionAdapter(
            Mock(get_annotations=Mock(return_value=(None, None)))
        )
        plate_number_dict = {"img.jpg": []}
        images = [bytes(0)]
        uris = ["img.jpg"]
        expected = {"img.jpg": ["LP B 542"]}
        car_attributes = adapter.cloud_vision_fallback(plate_number_dict, images, uris)
        assert car_attributes == expected

    def test_cloud_vision_fallback_not_needed(self):
        adapter = DetectionAdapter(
            Mock(get_annotations=Mock(return_value=(None, None)))
        )
        plate_number_dict = {"img.jpg": ["LP B 542"]}
        images = [bytes(0)]
        uris = ["img.jpg"]
        expected = {"img.jpg": ["LP B 542"]}
        car_attributes = adapter.cloud_vision_fallback(plate_number_dict, images, uris)
        assert car_attributes == expected
