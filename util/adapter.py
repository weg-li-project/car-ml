import io
import os
from typing import List, Tuple, Dict

from util.annotation_provider import AnnotationProvider
from util.transforms import result_list

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from yolo_cnn.detect import main as alpr_yolo_cnn_main

cloud_vision_fallback_active = (
    True if os.getenv("CLOUD_VISION_FALLBACK", "False").lower() == "true" else False
)


def detect_car_attributes(
    images: List[bytes], uris: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """Adapter function for calling main detection methods and transforming the results
    into a format that conforms to our API Specification.

    Args:
        images (List[bytes]): Images loaded in memory.
        uris (List[str]): Cloud Storage uris of the images.
    Returns:
        Tuple with final results for license plate number, car make and car color.
    """

    plate_numbers_dict, car_brands_dict, car_colors_dict = alpr_yolo_cnn_main(
        uris=uris, imgs=images
    )

    print(f"detect_car_attributes method : {plate_numbers_dict}")

    if cloud_vision_fallback_active:
        plate_numbers_dict = cloud_vision_fallback(plate_numbers_dict, images, uris)

    return (
        result_list(plate_numbers_dict),
        result_list(car_brands_dict),
        result_list(car_colors_dict),
    )


def cloud_vision_fallback(
    plate_numbers_dict, images, uris, annotation_provider=AnnotationProvider()
):
    """Augment provided plate_numbers_dict with Cloud Vision fallback when
    - No valid detected license plate number for one image
    - No valid detected license plate numbers overall

    Args:
        plate_numbers_dict (Dict[str, List[str]]): HTTP request object.
        images (List[bytes]): Images loaded in memory.
        uris (List[str]): Cloud Storage uris of the images.
        annotation_provider (AnnotationProvider): Provider of Cloud Vision annotations.
    Returns:
        Updated plate_numbers_dict with detected license plate numbers from the Cloud Vision API.
    """

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
            object_annotations, text_annotations = annotation_provider.get_annotations(
                key if key.startswith("gs://") else image
            )
            license_plate_nos = recognize_license_plate(
                io.BytesIO(image), object_annotations, text_annotations
            )
            plate_numbers_dict[key] = license_plate_nos
    return plate_numbers_dict
