
import re
import pandas as pd

class LicensePlateCandidate():

    def __init__(self, text, object_ = None, city_IDs_df = pd.read_csv('city_ids.csv', delimiter=';')):
        self.text = text
        self.object_ = object_
        self.city_IDs_df = city_IDs_df

    def __city_ID_exists__(self, city_ID):
        return (self.city_IDs_df['Abk.'] == city_ID).any()

    def __split_part1__(self, part1, min_len, max_len):

        if len(part1) == 2 and min_len == 2:
            # split 1 - 1
            city_ID, driver_ID_letters = list(part1)

            return city_ID, driver_ID_letters

        if len(part1) == 3 and min_len <= 3:

            # split 1 - 2
            city_ID = part1[0]
            driver_ID_letters = part1[1:]

            # does city_ID exist in database?
            if self.__city_ID_exists__(city_ID):
                return city_ID, driver_ID_letters
            else:
                # split 2 - 1
                city_ID = part1[:2]
                driver_ID_letters = part1[2:]

                return city_ID, driver_ID_letters

        if len(part1) == 4 and max_len >= 4:

            # split 2 - 2
            city_ID = part1[:2]
            driver_ID_letters = part1[2:]

            # does city_ID exist in database?
            if self.__city_ID_exists__(city_ID):
                return city_ID, driver_ID_letters
            else:
                # split 3 - 1
                city_ID = part1[:3]
                driver_ID_letters = part1[3:]

                return city_ID, driver_ID_letters

        if len(part1) == 5 and max_len >= 4:
            # split 3 - 2
            city_ID = part1[:2]
            driver_ID_letters = part1[2:]

            return city_ID, driver_ID_letters

    def __split_1_blank__(self, license_plate_no):

        part1, part2 = license_plate_no.split()

        if part2.isdigit():
            driver_ID_digits = part2
            city_ID, driver_ID_letters = self.__split_part1__(part1, 2, 5)
        else:
            city_ID = part1

            # split at first digit
            m = re.search(r"\d", part2)
            driver_ID_letters = part2[:m.start()]
            driver_ID_digits = part2[m.start():]

        # all signs after blank are digits?
        if not driver_ID_digits.isdigit():
            return license_plate_no, False, 'signs after blank are not only digits'

        license_plate_no = city_ID + ' ' + driver_ID_letters + ' ' + driver_ID_digits + self.last_sign

        # does city_ID exist in database?
        return license_plate_no, self.__city_ID_exists__(city_ID), 'city_ID does not exist in the database'

    def __split_no_blanks__(self, license_plate_no):

        if len(license_plate_no) == 3:

            city_ID, driver_ID_letters, driver_ID_digits = list(license_plate_no)

        else:

            if len(license_plate_no) == 4:
                min_len, max_len = 2, 3

            if len(license_plate_no) == 5:
                min_len, max_len = 2, 4

            if len(license_plate_no) == 6:
                min_len, max_len = 2, 5

            if len(license_plate_no) == 7:
                min_len, max_len = 3, 5

            if len(license_plate_no) == 8:
                min_len, max_len = 4, 5

            # split license_plate_no after first occurrence of digit
            m = re.search(r"\d", license_plate_no)
            part1 = license_plate_no[:m.start()]
            part2 = license_plate_no[m.start():]

            driver_ID_digits = part2

            city_ID, driver_ID_letters = self.__split_part1__(part1, min_len, max_len)

        # all signs after blank are digits?
        if not driver_ID_digits.isdigit():
            return license_plate_no, False, 'signs after blank are not only digits'

        # does city_ID only consist of letters?
        if not city_ID.isalpha():
            return license_plate_no, False, 'signs before blank are not only letters'

        # does driver_ID_letters only contain letters?
        if not driver_ID_letters.isalpha():
            return license_plate_no, False, 'driver_ID_letters does contain digits'

        license_plate_no = city_ID + ' ' + driver_ID_letters + ' ' + driver_ID_digits + self.last_sign

        # does city_ID exist in database?
        return license_plate_no, self.__city_ID_exists__(city_ID), 'city_ID does not exist in the database'

    def checkCandidate(self):

        self.last_sign = ''

        # remove all blanks at the beginning and end
        license_plate_no = self.text.strip()

        # is the last sign an 'E' (electro vehicles) or a 'H' (olt timer)
        if license_plate_no[-1] == 'H' or license_plate_no[-1] == 'E':
            self.last_sign = license_plate_no[-1]
            license_plate_no = license_plate_no[:-1]

        # does the license_plate_no end with a digit?
        if not license_plate_no[-1].isdigit():
            return license_plate_no, False, 'license_plate_no does not end with digit after removing and is no EV or old timer'

        # does the license_plate_no contain 4 digits max?
        if len(''.join(x for x in license_plate_no if x.isdigit())) > 4:
            return license_plate_no, False, 'license_plate_no does contain more than 4 digits'

        # does the license_plate_no start with two letters?
        if not license_plate_no[0].isalpha() and not license_plate_no[1].isalpha():
            return license_plate_no, False, 'license_plate_no does not start with 2 letters'

        # are all letters upper case?
        if not ''.join(x for x in license_plate_no if x.isalpha()).isupper():
            return license_plate_no, False, 'some letters are not upper case'

        # does the license_plate_no contain blanks?
        num_blanks = len(''.join(x for x in license_plate_no if x.isspace()))

        if num_blanks >= 3:
            return license_plate_no, False, 'license_plate_no contains more than 2 blanks after removing leading and trailing white spaces'

        if num_blanks == 2:

            # does the license_plate_no has the correct length?
            if len(license_plate_no) > 10 or len(license_plate_no) < 5:
                return license_plate_no, False, 'license_plate_no does not have the correct length'

            city_ID, driver_ID_letters, driver_ID_digits = license_plate_no.split()

            # does driver_ID_letters only contain letters?
            if not driver_ID_letters.isalpha():
                return license_plate_no, False, 'driver_ID_letters does contain digits'

            # does driver_ID_digits only contain digits?
            if not driver_ID_digits.isdigit():
                return license_plate_no, False, 'driver_ID_digits does not only contain digits'

            # does city_ID exist in database?
            return license_plate_no, self.__city_ID_exists__(city_ID), 'city_ID does not exist in the database'

        if num_blanks == 1:

            # does the license_plate_no has the correct length?
            if len(license_plate_no) > 10 or len(license_plate_no) < 5:
                return license_plate_no, False, 'license_plate_no does not have the correct length'

            return self.__split_1_blank__(license_plate_no)

        if num_blanks == 0:

            # does the license_plate_no has the correct length?
            if len(license_plate_no) < 3 or len(license_plate_no) > 8:
                return license_plate_no, False, 'license_plate_no does not have the correct length'

            return self.__split_no_blanks__(license_plate_no)
