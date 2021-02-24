import unittest

from util.transforms import (
    to_json_suggestions,
    get_uniques,
    order_by_frequency,
    make_list,
)


class MyTestCase(unittest.TestCase):
    def test_to_json_suggestions_empty(self):
        suggestions = (
            '{"suggestions": {"license_plate_number": [], "make": [], "color": []}}'
        )
        json_suggestions: str = to_json_suggestions()
        self.assertEqual(json_suggestions, suggestions)

    def test_to_json_suggestions_with_data(self):
        suggestions = '{"suggestions": {"license_plate_number": ["HM VM 546"], "make": ["BMW"], "color": ["blue"]}}'
        json_suggestions: str = to_json_suggestions(
            license_plate_numbers=["HM VM 546"], makes=["BMW"], colors=["blue"]
        )
        self.assertEqual(json_suggestions, suggestions)

    def test_get_uniques(self):
        expected = ["6", "8", "10", "4"]
        seq = ["6", "8", "6", "10", "10", "4", "6"]

        actual = get_uniques(seq)

        self.assertListEqual(expected, actual)

    def test_order_by_frequency(self):
        expected = ["6", "6", "6", "10", "10", "4", "4", "8"]
        seq = ["6", "8", "6", "10", "10", "4", "6", "4"]

        actual = order_by_frequency(seq)

        self.assertListEqual(expected, actual)

    def test_to_list(self):
        d = {"0": [4, 5], "@": [5, 8], "p": [2, 0, 1]}
        expected = [4, 5, 5, 8, 2, 0, 1]

        actual = make_list(d)

        self.assertListEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
