#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''Compute the average output of the four detectors of a polarimeter

This script loads the data acquired during a test and outputs the statistics of
the four PWR outputs. The output is printed in JSON format, and it is suitable
for inclusion in the test database
'''

import numpy as np
from argparse import ArgumentParser
import simplejson as json
import sys


def parse_command_line():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('text_file', type=str,
                        help='''Name of the text file containing the data 
                        acquired during the test''')
    return parser.parse_args()


def main():
    args = parse_command_line()
    data = np.loadtxt(args.text_file, skiprows=1)
    output = {
        'detector_offsets': {
            'PWR0_adu': float(np.round(np.mean(data[:, 7]), decimals=1)),
            'PWR1_adu': float(np.round(np.mean(data[:, 8]), decimals=1)),
            'PWR2_adu': float(np.round(np.mean(data[:, 9]), decimals=1)),
            'PWR3_adu': float(np.round(np.mean(data[:, 10]), decimals=1)),
        }
    }

    json.dump(output, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
