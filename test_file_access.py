# -*- encoding: utf-8 -*-

import os.path

from unittest import TestCase
import file_access as fa

TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              'testdata'))


class TestTimestreamCreation(TestCase):
    def testHdf5Load(self):
        metadata, data = fa.load_hdf5_file(os.path.join(
            TEST_DATA_PATH, 'hdf5_timestream.h5'))
        self.assertIsInstance(data, fa.Timestream)

    def testTextLoad(self):
        data = fa.load_text_file(os.path.join(
            TEST_DATA_PATH, 'text_file.txt'))
        self.assertIsInstance(data, fa.Timestream)


class CheckContentsMixin:
    def testValues(self):
        self.assertEqual(len(self.data.time_s), 25)

        self.assertEqual(self.data.time_s[0], 0.0)
        self.assertAlmostEqual(self.data.time_s[-1], 0.96)

        self.assertEqual(len(self.data.pctime), 25)

        self.assertEqual(self.data.pctime[0], 955)
        self.assertEqual(self.data.pctime[-1], 955)

        self.assertEqual(len(self.data.record), 25)

        self.assertEqual(self.data.record[0], 0)
        self.assertEqual(self.data.record[-1], 0)

        self.assertEqual(self.data.demodulated.shape, (25, 4))

        self.assertEqual(self.data.demodulated[0, 0], 7)
        self.assertEqual(self.data.demodulated[-1, 0], 3)

        self.assertEqual(self.data.demodulated[0, 1], 0)
        self.assertEqual(self.data.demodulated[-1, 1], -3)

        self.assertEqual(self.data.demodulated[0, 2], 12)
        self.assertEqual(self.data.demodulated[-1, 2], 6)

        self.assertEqual(self.data.demodulated[0, 3], 14)
        self.assertEqual(self.data.demodulated[-1, 3], 8)

        self.assertEqual(self.data.power.shape, (25, 4))

        self.assertEqual(self.data.power[0, 0], 63930)
        self.assertEqual(self.data.power[-1, 0], 63978)

        self.assertEqual(self.data.power[0, 1], 62631)
        self.assertEqual(self.data.power[-1, 1], 62643)

        self.assertEqual(self.data.power[0, 2], 61563)
        self.assertEqual(self.data.power[-1, 2], 61576)

        self.assertEqual(self.data.power[0, 3], 62753)
        self.assertEqual(self.data.power[-1, 3], 62791)

        self.assertEqual(len(self.data.rfpower_db), 25)

        self.assertEqual(self.data.rfpower_db[0], -1)
        self.assertEqual(self.data.rfpower_db[-1], -1)

        self.assertEqual(len(self.data.freq_hz), 25)

        self.assertEqual(self.data.freq_hz[0], -1)
        self.assertEqual(self.data.freq_hz[-1], -1)


class TestHdf5TimestreamContents(TestCase, CheckContentsMixin):
    def setUp(self):
        self.metadata, self.data = fa.load_hdf5_file(os.path.join(
            TEST_DATA_PATH, 'hdf5_timestream.h5'))


class TestTextTimestreamContents(TestCase, CheckContentsMixin):
    def setUp(self):
        self.data = fa.load_text_file(os.path.join(
            TEST_DATA_PATH, 'text_file.txt'))
