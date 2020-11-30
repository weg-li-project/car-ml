
import unittest

from license_plate_candidate import LicensePlateCandidate

class TestLicensePlateCandidate(unittest.TestCase):

    def testCheckCandidateRemovedLeadingWhiteSpaces(self):
        text = ' B MW 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        # TODO: assert

    def testCheckCandidateRemovedTrailingWhiteSpaces(self):
        text = 'B MW 1234 '
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        # TODO: assert

    def testCheckCandidateEV(self):
        text = 'B MW 1234E'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'B MW 1234E', msg
        assert res, 'testCheckCandidateEV failed'

    def testCheckCandidateOT(self):
        text = 'B MW 1234H'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'B MW 1234H', msg
        assert res, 'testCheckCandidateOT failed'

    def testCheckCandidateLastSignDigit(self):
        text = 'B MW 123A'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateLastSignDigit2(self):
        text = 'B MW 123E'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'B MW 123E', 'testCheckCandidateLastSignDigit2 failed'
        assert res, msg

    def testCheckCandidateLastSignDigit3(self):
        text = 'BE MW 1234E'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateMoreThan4Digits(self):
        text = 'B MW 12345'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateFirstTwoSignsLetters1(self):
        text = '1 MW 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateFirstTwoSignsLetters2(self):
        text = 'B 1W 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateFirstTwoSignsLetters3(self):
        text = '1 1W 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateAllLettersUpper(self):
        text = 'B Mw 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateMoreThan2Blanks(self):
        text = 'B M W 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidateWrongLength(self):
        text = 'IGB MW 1234'
        lpc = LicensePlateCandidate(text)
        _, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidate3Blanks(self):
        text = 'B A B 123'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert not res, msg

    def testCheckCandidate2Blanks(self):
        text = 'IGB A 123'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == text, 'testCheckCandidate2Blanks failed'
        assert res, msg

    def testCheckCandidate1Blank1(self):
        text = 'BAB 123'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'B AB 123', 'testCheckCandidate1Blank1 failed'
        assert res, msg

    def testCheckCandidate1Blank2(self):
        text = 'IGBA 123' # there is no city IG
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'IGB A 123', 'testCheckCandidate1Blank2 failed'
        assert res, msg

    def testCheckCandidate1Blank3(self):
        text = 'BA BA123'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'BA BA 123', 'testCheckCandidate1Blank3 failed'
        assert res, msg

    def testCheckCandidateNoBlanks1(self):
        text = 'BMW1234'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'B MW 1234', 'testCheckCandidateNoBlanks1 failed'
        assert res, msg

    def testCheckCandidateNoBlanks2(self):
        text = 'IGBA123'
        lpc = LicensePlateCandidate(text)
        license_plate_no, res, msg = lpc.checkCandidate()
        assert license_plate_no == 'IGB A 123', 'testCheckCandidateNoBlanks2 failed'
        assert res, msg

    def testCity_ID_exists1(self):
        text = 'B AB 123'
        lpc = LicensePlateCandidate(text)
        city_ID = 'B'
        _, res, msg = lpc.checkCandidate()
        assert lpc.__city_ID_exists__(city_ID), 'testCity_ID_exists1 failed'
        assert res, msg

    def testCity_ID_exists2(self):
        text = 'WND A 1234'
        lpc = LicensePlateCandidate(text)
        city_ID = 'WND'
        _, res, msg = lpc.checkCandidate()
        assert lpc.__city_ID_exists__(city_ID), 'testCity_ID_exists2 failed'
        assert res, msg

    def testCity_ID_exists3(self):
        text = 'ZZZ A 1234'
        lpc = LicensePlateCandidate(text)
        city_ID = 'ZZZ'
        _, res, msg = lpc.checkCandidate()
        assert not lpc.__city_ID_exists__(city_ID), 'testCity_ID_exists3 failed'
        assert not res, msg

    def testCity_ID_exists4(self):
        text = 'IGBA 123'
        lpc = LicensePlateCandidate(text)
        city_ID = 'IG'
        _, res, msg = lpc.checkCandidate()
        assert not lpc.__city_ID_exists__(city_ID), 'testCity_ID_exists4 failed'

if __name__ == '__main__':
    unittest.main()
