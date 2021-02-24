from unittest.mock import patch

from util.adapter import detect_car_attributes

plate_no_1 = "HH MC 1210"
plate_no_2 = "PI BA 2755"
plate_number_dict = {"/0.jpg": [plate_no_1, plate_no_2], "/1.jpg": [plate_no_2]}


class TestAdapter:
    @patch("util.adapter.alpr_yolo_cnn_main", return_value=(plate_number_dict, {}, {}))
    def test_get_car_attributes_suggestions(self, mock1):
        car_attributes_suggestions = detect_car_attributes([])
        assert car_attributes_suggestions == ([plate_no_2, plate_no_1], [], [])
