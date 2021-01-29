import os
from unittest.mock import patch

from util.alpr_adapter import recognize_license_plate_numbers

plate_no_1 = "HH MC 1210"
plate_no_2 = "PI BA 2755"
plate_number_dict = {
    '/0.jpg': [plate_no_1, plate_no_2],
    '/1.jpg': [plate_no_2]
}


class TestAlprAdapter:
    @patch('util.alpr_adapter.alpr_yolo_cnn_main', return_value=plate_number_dict)
    def test_get_license_plate_numbers(self, mock1):
        license_plate_nos = recognize_license_plate_numbers([])
        assert license_plate_nos == [plate_no_2, plate_no_1]