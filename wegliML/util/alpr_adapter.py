from typing import List

from pandas import np

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from alpr_yolo_cnn.detect import main as alpr_yolo_cnn_main


def recognize_license_plate_numbers(annotation_data) -> List[str]:
    images = [data[0] for data in annotation_data]
    plate_numbers_dict = alpr_yolo_cnn_main(images, cnn_advanced=False,
                                            yolo_checkpoint='./alpr_gcloud_vision/checkpoints/yolov4',
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
            for image, object_annotations, text_annotations in annotation_data:
                if image == img_path:
                    license_plate_nos = recognize_license_plate(image, object_annotations, text_annotations)
                    plate_numbers_dict[key] = license_plate_nos

    # remove duplicates
    license_plate_numbers_set = {}
    for l in plate_numbers_dict.values():
        license_plate_numbers_set = license_plate_numbers_set.union(set(l))
    return license_plate_numbers_set
