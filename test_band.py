#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import bandwidth as bw
from unittest import TestCase
import numpy as np


class TestBWCFfunction(TestCase):
    def test_squaredband(self):
        bandwidth = 7.0
        central_frequency = 43.0
        min_freq = 38.0
        max_freq = 50.0
        num_of_points = 121

        nu = np.linspace(min_freq, max_freq, num_of_points)
        band = np.zeros(num_of_points)
        band[np.abs(nu - central_frequency) <= bandwidth / 2] = 1.0

        computed_cfreq, computed_bwidth = bw.get_central_nu_bandwidth(nu, band)
        self.assertAlmostEqual(computed_cfreq, central_frequency)
        self.assertTrue(
            np.abs(computed_bwidth - bandwidth)
            < 2 * (max_freq - min_freq) / num_of_points
        )

    def test_triangularband(self):
        min_freq = 38.0
        max_freq = 50.0
        num_of_points = 121

        nu = np.linspace(min_freq, max_freq, num_of_points)
        band = np.zeros(num_of_points)
        band = (nu - min_freq) / (max_freq - min_freq)
        computed_cfreq, computed_bwidth = bw.get_central_nu_bandwidth(nu, band)

        self.assertAlmostEqual(computed_cfreq, (min_freq + 2 * max_freq) / 3, places=1)
