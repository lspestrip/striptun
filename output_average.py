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
    parser.add_argument('--time-interval', type=str, default=None,
                        help='''Start and end time of the interval to use
                        in the computation, in seconds. The two numbers
                        must be separated by an hyphen ("-").''')
    parser.add_argument('text_file', type=str,
                        help='''Name of the text file containing the data 
                        acquired during the test''')
    return parser.parse_args()


def main():
    args = parse_command_line()
    _, data = load_timestream(args.text_file)

    if args.time_interval:
        time0, time1 = [float(x) for x in args.time_interval.split('-')]
        mask = (data.time_s >= time0) & (data.time_s <= time1)
    else:
        mask = np.ones(data.power.shape[0], dtype='bool')

    output = {
        'detector_outputs': {
            'nsamples': len(data.power[mask, 0]),
            'time0': float(np.round(data.time_s[mask][0], 1)),
            'time1': float(np.round(data.time_s[mask][-1], 1)),
            'q1_adu': float(np.round(np.median(data.power[mask, 0]), decimals=1)),
            'u1_adu': float(np.round(np.median(data.power[mask, 1]), decimals=1)),
            'u2_adu': float(np.round(np.median(data.power[mask, 2]), decimals=1)),
            'q2_adu': float(np.round(np.median(data.power[mask, 3]), decimals=1)),
        }
    }

    json.dump(output, sys.stdout, indent=4)


if __name__ == '__main__':
    main()
