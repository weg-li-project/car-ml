import os
from typing import Final

from locust import HttpUser, task, tag

IMAGES: Final = {
    "small": [
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20191027_102538_size_small_car_two.jpg",
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20190811_095743_size_small_car_three.jpg",
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20190811_095734_size_small_car_four.jpg"
    ],
    "big": [
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20191129_085112_size_big_car_one.jpg",
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20201030_150403_size_big_car_three.jpg",
        f"gs://{os.environ['WEGLI_IMAGES_BUCKET_NAME']}/test/IMG_20201026_083127_size_big_car_multiple.jpg"
    ]
}


class ImageAnalysisUser(HttpUser):
    @tag("small", "one")
    @task(2)
    def one_small_image(self):
        body = {"google_cloud_urls": [IMAGES["small"][0]]}
        self.client.post("/", json=body)

    @tag("big", "one")
    @task(2)
    def one_big_image(self):
        body = {"google_cloud_urls": [IMAGES["big"][0]]}
        self.client.post("/", json=body)

    @tag("small", "two")
    @task(3)
    def two_small_images(self):
        body = {"google_cloud_urls": [*IMAGES["small"][0:2]]}
        self.client.post("/", json=body)

    @tag("big", "two")
    @task(3)
    def two_big_images(self):
        body = {"google_cloud_urls": [*IMAGES["big"][0:2]]}
        self.client.post("/", json=body)

    @tag("small", "three")
    @task(1)
    def three_small_images(self):
        body = {"google_cloud_urls": [*IMAGES["small"]]}
        self.client.post("/", json=body)

    @tag("big", "three")
    @task(1)
    def three_big_images(self):
        body = {"google_cloud_urls": [*IMAGES["big"]]}
        self.client.post("/", json=body)
