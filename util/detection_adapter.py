import io
import os
from typing import List, Tuple, Dict

from util.transforms import result_list

from alpr_gcloud_vision.alpr.license_plate_candidate import LicensePlateCandidate
from alpr_gcloud_vision.alpr.license_plate_recognition import recognize_license_plate
from yolo_cnn.detect import main as alpr_yolo_cnn_main

CLOUD_VISION_FALLBACK_ACTIVE = (
    True if os.getenv("CLOUD_VISION_FALLBACK", "False").strip().lower() == "true" else False
)


class DetectionAdapter:
    def __init__(
        self,
        annotation_provider,
        cloud_vision_fallback_active=CLOUD_VISION_FALLBACK_ACTIVE,
    ):
        self.annotation_provider = annotation_provider
        self.cloud_vision_fallback_active = cloud_vision_fallback_active

    def detect_car_attributes(
        self, images: List[bytes], uris: List[str]
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

        if self.cloud_vision_fallback_active:
            plate_numbers_dict = self.cloud_vision_fallback(
                plate_numbers_dict, images, uris
            )

        return (
            result_list(plate_numbers_dict),
            result_list(car_brands_dict),
            result_list(car_colors_dict),
        )

    def cloud_vision_fallback(self, plate_numbers_dict, images, uris):
        """Augment provided plate_numbers_dict with Cloud Vision fallback when
        - No valid detected license plate number for an image
        - No valid detected license plate numbers overall

        Args:
            plate_numbers_dict (Dict[str, List[str]]): HTTP request object.
            images (List[bytes]): Images loaded in memory.
            uris (List[str]): Cloud Storage uris of the images.
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
                (
                    object_annotations,
                    text_annotations,
                ) = self.annotation_provider.get_annotations(
                    key if key.startswith("gs://") else image
                )
                license_plate_nos = recognize_license_plate(
                    io.BytesIO(image), object_annotations, text_annotations
                )
                plate_numbers_dict[key] = license_plate_nos
        return plate_numbers_dict
