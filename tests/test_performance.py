import os
import unittest
from csv import DictReader

from tests.skip_markers import needs_private_testdata
from util.adapter import detect_car_attributes
from util.paths import (
    charges_csv_filepath,
    charges_schroeder_path,
)


def read_charges():
    with open(charges_csv_filepath, "r") as read_obj:
        charges = list(DictReader(read_obj))
        return list(
            map(
                lambda x: (
                    os.path.join(charges_schroeder_path, x["photos"]),
                    x["registration"],
                    x["brand"],
                    x["color"],
                ),
                charges,
            )
        )


def map_color(color):
    if color == "silver" or color == "gray":
        return "gray_silver"
    if color == "gold" or color == "yellow":
        return "gold_yellow"
    if color == "pink" or color == "purple" or color == "violet":
        return "pink_purple_violet"
    return color


@needs_private_testdata
class TestPerformance(unittest.TestCase):
    def test_color_conversion(self):
        colors = ["silver", "yellow", "pink", "violet", "red", "blue"]
        converted_colors = list(map(map_color, colors))
        assert converted_colors == [
            "gray_silver",
            "gold_yellow",
            "pink_purple_violet",
            "pink_purple_violet",
            "red",
            "blue",
        ]

    def test_detection_performance(self):
        charges = read_charges()

        correct_license_plate_numbers = 0
        correct_makes = 0
        correct_colors = 0
        for image_path, license_plate_number, make, color in charges:
            with open(image_path, "rb") as image:
                filename = str(image_path.split("/")[-1])
                img = bytes(bytearray(image.read()))
                try:
                    (
                        license_plate_numbers,
                        car_brands,
                        car_colors,
                    ) = detect_car_attributes(
                        [(filename, img)],
                        cloud_vision_fallback_active=True
                        if os.getenv("CLOUD_VISION_FALLBACK", "False").lower() == "true"
                        else False,
                    )

                    if license_plate_number in license_plate_numbers:
                        correct_license_plate_numbers += 1
                    else:
                        print(
                            f"Incorrect lpn for {filename}; Detected: {license_plate_numbers} - Actual: {license_plate_number}"
                        )

                    if make in car_brands:
                        correct_makes += 1
                    else:
                        print(
                            f"Incorrect make for {filename}; Detected: {car_brands} - Actual: {make}"
                        )

                    if map_color(color) in car_colors:
                        correct_colors += 1
                    else:
                        print(
                            f"Incorrect color for {filename}; Detected: {car_colors} - Actual: {color}"
                        )

                except Exception as err:
                    print(f"Error for {filename}; Message: {err}")

        num_charges = len(charges)

        precision_license_plate_numbers = correct_license_plate_numbers / num_charges
        print(f"LPN detection precision: {precision_license_plate_numbers}")

        precision_makes = correct_makes / num_charges
        print(f"Make detection precision: {precision_makes}")

        precision_colors = correct_colors / num_charges
        print(f"Color detection precision: {precision_colors}")

        self.assertTrue(precision_license_plate_numbers > 0.38)
        self.assertTrue(precision_makes > 0.90)
        self.assertTrue(precision_colors > 0.75)
