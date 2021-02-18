import io
from typing import List, Tuple, Dict

from alpr_gcloud_vision.core.annotations import get_annotations
from util.transforms import order_by_frequency, get_uniques, to_list

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from yolo_cnn.detect import main as alpr_yolo_cnn_main


def detect_car_attributes(
    image_data: List[Tuple[str, bytes]], cloud_vision_fallback_active: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    uris = [data[0] for data in image_data]
    images = [data[1] for data in image_data]

    plate_numbers_dict, car_brands_dict, car_colors_dict = alpr_yolo_cnn_main(
        uris=uris, images=images
    )

    if False:
        plate_numbers_dict = cloud_vision_fallback(plate_numbers_dict, images, uris)

    return (
        result_list(plate_numbers_dict),
        result_list(car_brands_dict),
        result_list(car_colors_dict),
    )


def cloud_vision_fallback(plate_numbers_dict, images, uris):
    # Check if any found license plate number is a valid license plate
    for key in plate_numbers_dict.keys():
        results = []
        for lp in plate_numbers_dict[key]:
            lpc = LicensePlateCandidate(lp)
            license_plate_no, res, msg = lpc.checkCandidate()
            results.append(res)

        # Cloud vision fallback if no lpn or potential candidate found
        if not plate_numbers_dict[key] or not any(results):
            image = images[uris.index(key)]
            object_annotations, text_annotations = get_annotations(image)
            license_plate_nos = recognize_license_plate(
                io.BytesIO(image), object_annotations, text_annotations
            )
            plate_numbers_dict[key] = license_plate_nos
    return plate_numbers_dict


def result_list(d: Dict[str, List[str]]) -> List[str]:
    return get_uniques(order_by_frequency(to_list(d)))
