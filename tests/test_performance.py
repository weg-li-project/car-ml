import os
import unittest
from csv import DictReader

from tqdm import tqdm

from tests.skip_markers import needs_private_testdata
from util.transforms import get_uniques
from yolo_cnn.detect import load_models, detect_recognize_car, detect_recognize_plate

from util.paths import yolo_lp_model_path, yolo_car_model_path, cnn_alpr_model_path, cnn_color_rec_model_path, \
    cnn_car_rec_model_path, charges_csv_filepath, charges_schroeder_path


def load():
    return load_models(
        yolo_lp_model_path=yolo_lp_model_path,
        yolo_car_model_path=yolo_car_model_path,
        cnn_alpr_checkpoint_path=cnn_alpr_model_path,
        cnn_color_rec_checkpoint_path=cnn_color_rec_model_path,
        cnn_car_rec_checkpoint_path=cnn_car_rec_model_path
    )


def read_charges(column_name: str):
    with open(charges_csv_filepath, 'r') as read_obj:
        charges = list(DictReader(read_obj))
        return list(map(lambda x: (os.path.join(charges_schroeder_path, x['photos']), x[column_name]), charges))


@needs_private_testdata
class MyTestCase(unittest.TestCase):
    def test_detect_car_make(self):
        yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec = load()
        charges = read_charges('brand')

        correct = 0
        for image_path, make in tqdm(charges, 'Car makes'):
            with open(image_path, "rb") as image:
                print(image_path)
                img = bytearray(image.read())
                try:
                    (car_brands,) = detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, image_path, img, yolo_car)
                    if make in car_brands[:3]:
                        correct += 1
                        print('Correct prediction')
                    print(f'Car makes detected: {car_brands}, actual car make: {make}')
                except Exception as err: #TypeError (cannot unpack non-iterable NoneType object), ValueError (not enough values to unpack) -> analyze_box
                    print(err)

        precision = correct / len(charges)
        print(f'Make precision: {precision}')
        self.assertTrue(precision > 0.85)

    def test_detect_car_color(self):
        yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec = load()
        charges = read_charges('color')

        correct = 0
        for image_path, color in tqdm(charges, 'Car colors'):
            with open(image_path, "rb") as image:
                print(image_path)
                img = bytearray(image.read())
                try:
                    (_, car_colors) = detect_recognize_car(cnn_alpr, cnn_car_rec, cnn_color_rec, image_path, img, yolo_car)
                    if color in car_colors[:3]:
                        correct += 1
                        print('Correct prediction')
                    print(f'Car colors detected: {car_colors}, actual car color: {color}')
                except Exception as err:
                    print(err)

        precision = correct / len(charges)
        print(f'Color precision: {precision}')
        self.assertTrue(precision > 0.85)

    def test_detect_car_license_plate_number(self):
        yolo_lp, yolo_car, cnn_alpr, cnn_color_rec, cnn_car_rec = load()
        charges = read_charges('registration')

        correct = 0
        for image_path, license_plate_number in tqdm(charges, 'License plate numbers'):
            with open(image_path, "rb") as image:
                print(image_path)
                img = bytearray(image.read())
                try:
                    license_plate_numbers = get_uniques([lpn for lpns in detect_recognize_plate(cnn_alpr, cnn_car_rec, cnn_color_rec, image_path, img, yolo_lp).values() for lpn in lpns])
                    if license_plate_number in license_plate_numbers[:3]:
                        correct += 1
                        print('Correct prediction')
                    print(f'Car license plate numbers detected: {license_plate_numbers}, actual car license plate number: {license_plate_number}')
                except Exception as err:
                    print(err)

        precision = correct / len(charges)
        print(f'ALPR precision: {precision}')
        self.assertTrue(precision > 0.75)
