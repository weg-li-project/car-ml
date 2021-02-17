import io
from typing import List, Tuple

from alpr_gcloud_vision.core.annotations import get_annotations
from util.transforms import order_by_frequency, get_uniques

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from yolo_cnn.detect import main as alpr_yolo_cnn_main


def detect_car_attributes(
    image_data: List[Tuple[str, bytes]]
) -> Tuple[List[str], List[str], List[str]]:
    uris = [data[0] for data in image_data]
    images = [data[1] for data in image_data]
    plate_numbers_dict, car_brands_dict, car_colors_dict = alpr_yolo_cnn_main(
        uris=uris, images=images
    )

    # check if any found license plate number is a valid license plate
    for key in plate_numbers_dict.keys():
        results = []
        for lp in plate_numbers_dict[key]:
            lpc = LicensePlateCandidate(lp)
            license_plate_no, res, msg = lpc.checkCandidate()
            results.append(res)

        if plate_numbers_dict[key] is None or plate_numbers_dict[key] == []:
            img_path = key
            image = images[uris.index(img_path)]
            object_annotations, text_annotations = get_annotations(image)
            license_plate_nos = recognize_license_plate(
                io.BytesIO(image), object_annotations, text_annotations
            )
            plate_numbers_dict[key] = license_plate_nos

    license_plate_numbers = [
        lpn for lpns in plate_numbers_dict.values() for lpn in lpns
    ]
    car_brands = [cb for cbs in car_brands_dict.values() for cb in cbs]
    car_colors = [cc for ccs in car_colors_dict.values() for cc in ccs]
    return (
        get_uniques(order_by_frequency(license_plate_numbers)),
        get_uniques(order_by_frequency(car_brands)),
        get_uniques(order_by_frequency(car_colors)),
    )
