from typing import List

import numpy as np

from alpr_gcloud_vision.core.annotations import get_annotations_from_gcs_uri
from util.transforms import order_by_frequency, get_uniques

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from alpr_yolo_cnn.detect import main as alpr_yolo_cnn_main


def recognize_license_plate_numbers(annotation_data) -> List[str]:
    uris = [data[0] for data in annotation_data]
    images = [data[1] for data in annotation_data]
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

        if plate_numbers_dict[key] is None or plate_numbers_dict[key] == [] or not (np.array(results)).any():
            img_path = key
            for uri, image, object_annotations, text_annotations in get_annotations_from_gcs_uri(img_path):
                if uri == img_path:
                    license_plate_nos = recognize_license_plate(image, object_annotations, text_annotations)
                    plate_numbers_dict[key] = license_plate_nos

    license_plate_numbers = [lpn for lpns in plate_numbers_dict.values() for lpn in lpns]
    return get_uniques(order_by_frequency(license_plate_numbers))
