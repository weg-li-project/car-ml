import unittest
import numpy as np
import pandas as pd
import json
from PIL import Image
from google.cloud.vision_v1.types.image_annotator import LocalizedObjectAnnotation, EntityAnnotation
from matplotlib.patches import Polygon

from alpr_gcloud_vision.alpr.object_detection import DetectedObject
from util.paths import vision_api_results_path, charges_schroeder_path


class TestObjectDetection(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestObjectDetection, self).__init__(*args, **kwargs)
        self.results_api_df = pd.read_csv(vision_api_results_path, delimiter=';')
        np.random.seed(3) # i = 664
        i = np.random.randint(self.results_api_df.shape[0])
        self.__init_objects__(i)
        self.__init_texts__(i)
        self.__init_img_path__(i)
        self.img = Image.open(self.img_path)

    def __init_objects__(self, i):

        objects_csv = json.loads(self.results_api_df.iloc[i, :]['localized_object_annotations'])
        self.objects = []

        for object_ in objects_csv:
            loa = LocalizedObjectAnnotation(object_)
            self.objects.append(loa)

    def __init_texts__(self, i):

        texts_csv = json.loads(self.results_api_df.iloc[i, :]['text_annotations'])
        self.texts = []

        for text in texts_csv:
            ea = EntityAnnotation(text)
            self.texts.append(ea)

    def __init_img_path__(self, i):
        self.img_path = charges_schroeder_path + self.results_api_df.iloc[i, :]['filename']

    def testDetectedObjectPolygon(self):

        i = np.random.randint(len(self.objects))

        width, height = self.img.size[:2]
        object_ = DetectedObject(self.objects[i], self.img)

        assert object_.llc[0] == object_.llc_normalized.x * width, 'testDetectedObjectPolygon failed'
        assert object_.llc[1] == object_.llc_normalized.y * height, 'testDetectedObjectPolygon failed'
        assert object_.lrc[0] == object_.lrc_normalized.x * width, 'testDetectedObjectPolygon failed'
        assert object_.lrc[1] == object_.lrc_normalized.y * height, 'testDetectedObjectPolygon failed'
        assert object_.urc[0] == object_.urc_normalized.x * width, 'testDetectedObjectPolygon failed'
        assert object_.urc[1] == object_.urc_normalized.y * height, 'testDetectedObjectPolygon failed'
        assert object_.ulc[0] == object_.ulc_normalized.x * width, 'testDetectedObjectPolygon failed'
        assert object_.ulc[1] == object_.ulc_normalized.y * height, 'testDetectedObjectPolygon failed'

    def testCalculatePolyAreaSquare(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.llc = (0, 5)
        detected_object.lrc = (5, 5)
        detected_object.urc = (5, 0)
        detected_object.ulc = (0, 0)

        area = detected_object._calculate_Poly_area()

        assert area == 5 ** 2, 'testCalculatePolyAreaSquare failed'

    def testCalculatePolyAreaRectangle(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.llc = (0, 4)
        detected_object.lrc = (6, 4)
        detected_object.urc = (6, 0)
        detected_object.ulc = (0, 0)

        area = detected_object._calculate_Poly_area()

        assert area == 4 * 6, 'testCalculatePolyAreaRectangle failed'

    def testCalculatePolyAreaTrapez(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.llc = (2, 2)
        detected_object.lrc = (5, 2)
        detected_object.urc = (7, 1)
        detected_object.ulc = (1, 1)

        area = detected_object._calculate_Poly_area()
        assert area == 3 + 0.5 + 1, 'testCalculatePolyAreaTrapez failed'

    def testCalculatePolyArea1(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.llc = (1, 4)
        detected_object.lrc = (4, 4)
        detected_object.urc = (4, 1)
        detected_object.ulc = (0, 0)

        area = detected_object._calculate_Poly_area()
        assert area == 12, 'testCalculatePolyArea failed'

    def testCalculatePolyArea2(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.llc = (11, 13)
        detected_object.lrc = (15, 16)
        detected_object.urc = (14, 10)
        detected_object.ulc = (8, 9)

        area = detected_object._calculate_Poly_area()
        assert area == 21, 'testCalculatePolyArea failed'

    def testContainsObject(self):

        detected_object = DetectedObject(self.objects[0], self.img)
        detected_object.ulc_x = 0
        detected_object.ulc_y = 0
        detected_object.urc_x = 6
        detected_object.urc_y = 0
        detected_object.lrc_x = 6
        detected_object.lrc_y = 4
        detected_object.llc_x = 0
        detected_object.llc_y = 4

        other = Polygon(((2,2), (4,2), (4,3), (2,3)))

        assert detected_object.containsObject(other), 'testContainsObject failed'

    def testFindTexts(self):

        license_plate = DetectedObject(self.objects[1], self.img)

        assert 'DRO 1812' == license_plate.findTexts(self.texts), 'testFindTexts failed'

    def testFindTextsMount(self):

        i = self.results_api_df.index[self.results_api_df['filename'] == 'IMG_20190811_095716.jpg'][0]

        objects_csv = json.loads(self.results_api_df.iloc[i, :]['localized_object_annotations'])
        objects = []

        for object_ in objects_csv:
            loa = LocalizedObjectAnnotation(object_)
            objects.append(loa)

        texts_csv = json.loads(self.results_api_df.iloc[i, :]['text_annotations'])
        texts = []

        for text in texts_csv:
            ea = EntityAnnotation(text)
            texts.append(ea)

        img_path = charges_schroeder_path + 'IMG_20190809_161740.jpg'
        img = Image.open(img_path)

        license_plate = DetectedObject(objects[2], img)

        assert 'ST W 80' == license_plate.findTexts(texts), 'testFindTextsMount failed'

if __name__ == '__main__':
    unittest.main()
