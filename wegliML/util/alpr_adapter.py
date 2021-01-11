import io
from typing import List, Tuple

from alpr_gcloud_vision.core.annotations import get_annotations
from util.transforms import order_by_frequency, get_uniques

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from alpr_yolo_cnn.detect import main as alpr_yolo_cnn_main


def recognize_license_plate_numbers(image_data: List[Tuple[str, bytes]]) -> List[str]:
    uris = [data[0] for data in image_data]
    images = [data[1] for data in image_data]
    plate_numbers_dict = alpr_yolo_cnn_main(uris=uris, images=images, cnn_advanced=False,
                                            yolo_checkpoint='./alpr_yolo_cnn/checkpoints/yolov4/',
                                            cnn_checkpoint='./alpr_yolo_cnn/checkpoints/cnn/training')

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
            license_plate_nos = recognize_license_plate(io.BytesIO(image), object_annotations, text_annotations)
            plate_numbers_dict[key] = license_plate_nos

    license_plate_numbers = [lpn for lpns in plate_numbers_dict.values() for lpn in lpns]
    return get_uniques(order_by_frequency(license_plate_numbers))
