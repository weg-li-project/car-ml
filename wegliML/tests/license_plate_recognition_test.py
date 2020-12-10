
import unittest
import json
import numpy as np
import pandas as pd
from google.cloud.vision_v1.types.image_annotator import EntityAnnotation, LocalizedObjectAnnotation
from termcolor import colored

from wegliML.src.license_plate_recognition import recognize_license_plate

import warnings

class TestLicensePlateRecognition(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLicensePlateRecognition, self).__init__(*args, **kwargs)
        self.charges_path = '../data/charges_Schroeder/'
        self.charges_df = pd.read_csv(self.charges_path + 'charges.csv', delimiter=',')
        self.results_api_df = pd.read_csv('../data/testdata/vision_api_results.csv', delimiter=';')

    def __check_correct_license_plate__(self, i):

        row = self.results_api_df.iloc[i, :]
        img_path = row['filename']
        objects_csv = json.loads(row['localized_object_annotations'])
        texts_csv = json.loads(row['text_annotations'])

        bool = False

        # check if img_path exists in charges_df
        if (self.charges_df['photos'] == img_path).any():

            objects = []
            texts = []

            for text in texts_csv:
                ea = EntityAnnotation(text)
                texts.append(ea)

            for object_ in objects_csv:
                loa = LocalizedObjectAnnotation(object_)
                objects.append(loa)

            warnings.simplefilter("ignore", ResourceWarning)
            license_plate_nos = recognize_license_plate(self.charges_path + img_path, objects, texts)

            idx = self.charges_df.index[self.charges_df['photos'] == img_path][0]

            for license_plate_no in license_plate_nos:
                if self.charges_df['registration'][idx] == license_plate_no:
                    bool = True
                    break

            return bool

    def testImg1(self):

            bool = self.__check_correct_license_plate__(0)
            assert bool, 'testImg1 failed'

    def testImg2(self):

            bool = self.__check_correct_license_plate__(1)
            assert bool, 'testImg3 failed'

    def testImg3(self):

        bool = self.__check_correct_license_plate__(2)
        assert bool, 'testImg3 failed'

    def testImg4(self):

            bool = self.__check_correct_license_plate__(3)
            assert bool, 'testImg4 failed'

    def testImg5(self):

            bool = self.__check_correct_license_plate__(4)
            assert bool, 'testImg5 failed'

    def testImg6(self):

        bool = self.__check_correct_license_plate__(5)
        assert bool, 'testImg6 failed'

    def testImg7(self):

        bool = self.__check_correct_license_plate__(6)
        assert bool, 'testImg7 failed'

    def testImg8(self):

        bool = self.__check_correct_license_plate__(7)
        assert bool, 'testImg8 failed'

    def testImg9(self):

        bool = self.__check_correct_license_plate__(8)
        assert bool, 'testImg9 failed'

    def testImg10(self):

        bool = self.__check_correct_license_plate__(9)
        assert bool, 'testImg10 failed'

    def testAllImages(self):
        total = 0
        green, red, zeros = 0, 0, 0
        res = np.zeros(shape=(self.results_api_df.shape[0],))

        for index, row in self.results_api_df.iterrows():
            img_path = row['filename']
            objects_csv = json.loads(row['localized_object_annotations'])
            texts_csv = json.loads(row['text_annotations'])

            # check if img_path exists in charges_df
            if (self.charges_df['photos'] == img_path).any():

                objects = []
                texts = []

                for text in texts_csv:
                    ea = EntityAnnotation(text)
                    texts.append(ea)

                for object_ in objects_csv:
                    loa = LocalizedObjectAnnotation(object_)
                    objects.append(loa)

                warnings.simplefilter("ignore", ResourceWarning)

                license_plate_nos = recognize_license_plate(self.charges_path + img_path, objects, texts)

                idx = self.charges_df.index[self.charges_df['photos'] == img_path][0]
                total += 1

                if len(license_plate_nos) == 0:
                    print(colored('recognized license_plate_no: ' + str([]) + '\t' + self.charges_df['registration'][idx] + ' is the actual license_plate_no' + '\t' + str(red), 'red'))
                    red += 1
                    zeros += 1
                else:
                    for license_plate_no in license_plate_nos:
                        if license_plate_no == self.charges_df['registration'][idx]:
                            res[index] = 1
                            green += 1
                            print(colored('recognized license_plate_no: ' + license_plate_no + '\t' + self.charges_df['registration'][idx] + ' is the actual license_plate_no' + '\t' + str(green), 'green'))
                    if res[index] == 0:
                        for license_plate_no in license_plate_nos:
                            red += 1
                            print(colored('recognized license_plate_no: ' + license_plate_no + '\t' + self.charges_df['registration'][idx] + ' is the actual license_plate_no' + '\t' + str(red), 'red'))

        print('percentage: {}'.format(np.sum(res) / total))
        assert np.sum(res) / total >= 0.5, 'percentage smaller than 50%'

if __name__ == '__main__':
    unittest.main()
