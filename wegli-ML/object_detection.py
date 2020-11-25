
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from google.cloud import vision

class DetectedObject():

    def __init__(self, object_, img):

        self.name = object_.name
        self.confidence = object_.score

        self.llc_normalized = object_.bounding_poly.normalized_vertices[0]
        self.lrc_normalized = object_.bounding_poly.normalized_vertices[1]
        self.urc_normalized = object_.bounding_poly.normalized_vertices[2]
        self.ulc_normalized = object_.bounding_poly.normalized_vertices[3]
        self.Polygon_normalized = Polygon(xy=[(self.llc_normalized.x, self.llc_normalized.y), (self.lrc_normalized.x, self.lrc_normalized.y),
                                              (self.urc_normalized.x, self.urc_normalized.y), (self.ulc_normalized.x, self.ulc_normalized.y)],
                                          fill=False, linewidth=3, edgecolor='r')

        self.img = img
        img_width, img_height = self.img.size[:2]
        self.Polygon = np.stack((self.Polygon_normalized.get_xy()[:, 0] * img_width, self.Polygon_normalized.get_xy()[:, 1]) * img_height, axis=1)
        self.llc = self.Polygon.get_xy()[0]
        self.lrc = self.Polygon.get_xy()[1]
        self.urc = self.Polygon.get_xy()[2]
        self.ulc = self.Polygon.get_xy()[3]
        self.Polygon_area = self.calculate_Poly_area()

    def calculate_Poly_area(self):

        a1 = math.dist(self.llc, self.lrc)
        b1 = math.dist(self.lrc, self.urc)
        c1 = math.dist(self.llc, self.urc)

        s1 = (a1 + b1 + c1) / 2
        triangle1_area = (s1 * (s1 - 1) * (s1 - b1) * (s1 - c1)) ** 0.5

        a2 = math.dist(self.llc, self.ulc)
        b2 = math.dist(self.ulc, self.urc)
        c2 = c1

        s2 = (a1 + b1 + c1) / 2
        triangle2_area = (s2 * (s2 - 1) * (s2 - b2) * (s2 - c2)) ** 0.5

        return triangle1_area + triangle2_area

    def isObject(self, object_name):
        return self.name == object_name

    def containsObject(self, other):
        # TODO: add tolerance
        if self.llc.x >= other.llc.x and self.lrc.x <= other.lrc.x and self.ulc.x >= other.ulc.x and self.urc.x <= other.urc.x and self.llc.y <= other.llc.y \
                and self.ulc.y >= other.urc.y and self.lrc.y <= other.lrc.y and self.urc.y >= other.urc.y:
            return True
        else:
            return False

    def isBigger(self, other):
        if self.Polygon_area > other.Polygon_area:
            return True
        else:
            return False

    def get_Polygon(self):
        return self.Polygon

    def get_normalized_Polygon(self):
        return self.Polygon_normalized

    def plot_Polygon_img(self, img):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        ax.imshow(img)
        ax.add_patch(self.Polygon)
        plt.show()

def localize_objects(img_path):
    client = vision.ImageAnnotatorClient()

    with open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    return objects

if __name__ == "__main__":

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'wegli-b95e0c13796c.json'
    SOURCE_PATH = 'imgs/'

    objects = localize_objects(SOURCE_PATH + 'test_img1.jpeg')

    detected_objects = []

    for object_ in objects:
        detected_objects.append(DetectedObject(object_))

    cars = []
    license_plates = []
    
    for object_ in detected_objects:
        if object_.isObject('Car'):
            cars.append(object_)
        if object_.isObject('License Plate'):
            license_plates.append(object_)
