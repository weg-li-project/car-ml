
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
        self.object_ = object_
        self.img = img
        self.ulc_normalized, self.urc_normalized, self.lrc_normalized, self.llc_normalized = self.__get_normalized_corners__()
        self.Polygon_normalized = self.__compute_normalized_Poly__()
        self.Polygon = self.__compute_Poly__()
        self.ulc, self.urc, self.lrc, self.llc = self.__get_corners__()
        self.Polygon_area = self.__calculate_Poly_area__()

    def __get_normalized_corners__(self):
        return self.object_.bounding_poly.normalized_vertices[0], self.object_.bounding_poly.normalized_vertices[1], \
               self.object_.bounding_poly.normalized_vertices[2], self.object_.bounding_poly.normalized_vertices[3]

    def __get_corners__(self):
        self.ulc_x = self.Polygon.get_xy()[0][0]
        self.ulc_y = self.Polygon.get_xy()[0][1]
        self.urc_x = self.Polygon.get_xy()[1][0]
        self.urc_y = self.Polygon.get_xy()[1][1]
        self.lrc_x = self.Polygon.get_xy()[2][0]
        self.lrc_y = self.Polygon.get_xy()[2][1]
        self.llc_x = self.Polygon.get_xy()[3][0]
        self.llc_y = self.Polygon.get_xy()[3][1]
        return self.Polygon.get_xy()[0], self.Polygon.get_xy()[1], self.Polygon.get_xy()[2], self.Polygon.get_xy()[3]

    def __compute_normalized_Poly__(self):
        return Polygon(xy=[(self.ulc_normalized.x, self.ulc_normalized.y), (self.urc_normalized.x, self.urc_normalized.y),
                    (self.lrc_normalized.x, self.lrc_normalized.y), (self.llc_normalized.x, self.llc_normalized.y)],
                fill=False, linewidth=3, edgecolor='r')

    def __compute_Poly__(self):
        img_width, img_height = self.img.size[:2]
        return Polygon(np.stack((self.Polygon_normalized.get_xy()[:, 0] * img_width, self.Polygon_normalized.get_xy()[:, 1] * img_height), axis=1))

    def __calculate_Poly_area__(self):
        a = 0.0;
        corners = [self.ulc, self.urc, self.lrc, self.llc]
        for i in range(4):
            a += (corners[i][1] + corners[(i+1) % 4][1]) * (corners[i][0] - corners[(i+1) % 4][0])
        return abs(a) / 2.0

    def isObject(self, object_name):
        return self.name == object_name

    def containsObject(self, other):

        if isinstance(other, DetectedObject):
            ulc_x = other.ulc.x
            ulc_y = other.ulc.y
            urc_x = other.urc.x
            urc_y = other.urc.y
            lrc_x = other.lrc.x
            lrc_y = other.lrc.y
            llc_x = other.llc.x
            llc_y = other.llc.y

        elif isinstance(other, Polygon):
            poly = other
            ulc_x = poly.get_xy()[0][0]
            ulc_y = poly.get_xy()[0][1]
            urc_x = poly.get_xy()[1][0]
            urc_y = poly.get_xy()[1][1]
            lrc_x = poly.get_xy()[2][0]
            lrc_y = poly.get_xy()[2][1]
            llc_x = poly.get_xy()[3][0]
            llc_y = poly.get_xy()[3][1]

        height, width = self.img.size[:2]

        # TODO: improve tolerance
        eps_x = (self.lrc_x - self.llc_x) * 0.15
        eps_y = (self.llc_y - self.ulc_y) * 0.15

        if self.llc_x <= llc_x + eps_x and self.lrc_x >= lrc_x - eps_x and self.urc_x >= urc_x - eps_x and self.ulc_x <= ulc_x + eps_x \
                and self.llc_y >= llc_y - eps_y and self.lrc_y >= lrc_y - eps_y and self.urc_y <= urc_y + eps_y and self.ulc_y <= ulc_y + eps_y:
            return True
        else:
            return False

    def findTexts(self, texts):
        containedTexts = []

        for text in texts:
            xy = [(text.bounding_poly.vertices[i].x, text.bounding_poly.vertices[i].y) for i in range(len(text.bounding_poly.vertices))]
            poly = Polygon(xy)
            if self.containsObject(poly):
                containedTexts.append((text.bounding_poly.vertices[0].x, text))

        # sort text list by llc.x
        containedTexts.sort(key=lambda text: text[0])

        output_text = [text[1].description for text in containedTexts]

        output_text = ' '.join(output_text)

        return output_text

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
