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
from file_access import load_timestream


def parse_command_line():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('text_file', type=str,
                        help='''Name of the text file containing the data 
                        acquired during the test''')
    return parser.parse_args()


def main():
    args = parse_command_line()
    _, data = load_timestream(args.text_file)
    output = {
        'detector_outputs': {
            'q1_adu': float(np.round(np.median(data.power[:, 0]), decimals=1)),
            'u1_adu': float(np.round(np.median(data.power[:, 1]), decimals=1)),
            'u2_adu': float(np.round(np.median(data.power[:, 2]), decimals=1)),
            'q2_adu': float(np.round(np.median(data.power[:, 3]), decimals=1)),
        }
    }

    json.dump(output, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
