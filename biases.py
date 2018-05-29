#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
import logging as log
import os.path
from shutil import copyfile
from typing import Any, Dict, List, Tuple

from json_save import save_parameters_to_json
from reports import create_report, get_code_version_params
import numpy as np

from file_access import load_metadata


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``input_url``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter (any text string
                                is ok, it is used in the reports)''')
    parser.add_argument('input_url', type=str,
                        help='''File/URL containing the data being saved''')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the path does not exist, it will be created''')
    return parser.parse_args()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def detector_type(metadata: Dict[str, Any]):
    """Return the type of detector (always 0 for Q, can be 1 or 2 for W)

    The detector type distinguishes among different ways to drive the HEMTs.
    """
    if metadata['band'] == 'Q':
        return 0

    if metadata['band'] != 'W':
        raise ValueError('Unknown band {0} for polarimeter STRIP{1:02d}'
                         .format(metadata['band'], metadata['polarimeter_number']))

    # We have a W-band polarimeter here
    biases = dotdict(metadata['hemt_biases'])
    # Convert everything in microV and perform all comparisons using integers
    ha2_vd = int(1e6 * biases.drain_voltage_ha2_V)
    ha3_vd = int(1e6 * biases.drain_voltage_ha3_V)
    ha2_vg = int(1e3 * biases.gate_voltage_ha2_mV)
    ha3_vg = int(1e3 * biases.gate_voltage_ha3_mV)

    hb2_vd = int(1e6 * biases.drain_voltage_hb2_V)
    hb3_vd = int(1e6 * biases.drain_voltage_hb3_V)
    hb2_vg = int(1e3 * biases.gate_voltage_hb2_mV)
    hb3_vg = int(1e3 * biases.gate_voltage_hb3_mV)

    if (ha2_vd == ha3_vd) and (ha2_vg == ha3_vg) and (hb2_vd == hb3_vd) and (hb2_vg == hb3_vg):
        return 1

    return 2


def build_dict_from_results(pol_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    biases = dotdict(metadata['hemt_biases'])

    return {
        'polarimeter': pol_name,
        'title': 'Bias configuration for polarimeter {0}'.format(pol_name),
        'ha1_vd': biases.drain_voltage_ha1_V * 1e3,
        'hb1_vd': biases.drain_voltage_hb1_V * 1e3,
        'ha2_vd': biases.drain_voltage_ha2_V * 1e3,
        'hb2_vd': biases.drain_voltage_hb2_V * 1e3,
        'ha3_vd': biases.drain_voltage_ha3_V * 1e3,
        'hb3_vd': biases.drain_voltage_hb3_V * 1e3,
        'ha1_id': biases.drain_current_ha1_mA,
        'hb1_id': biases.drain_current_hb1_mA,
        'ha2_id': biases.drain_current_ha2_mA,
        'hb2_id': biases.drain_current_hb2_mA,
        'ha3_id': biases.drain_current_ha3_mA,
        'hb3_id': biases.drain_current_hb3_mA,
        'ha1_vg': biases.gate_voltage_ha1_mV,
        'hb1_vg': biases.gate_voltage_hb1_mV,
        'ha2_vg': biases.gate_voltage_ha2_mV,
        'hb2_vg': biases.gate_voltage_hb2_mV,
        'ha3_vg': biases.gate_voltage_ha3_mV,
        'hb3_vg': biases.gate_voltage_hb3_mV,
        'band': metadata['band'],
        'detector_type': detector_type(metadata),
    }


def main():

    log.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=log.DEBUG)
    args = parse_arguments()

    log.info('Creating bias report for radiometer "%s"', args.polarimeter_name)
    log.info('Reading data from file "%s"', args.input_url)
    log.info('Writing the report into "%s"', args.output_path)

    # Create the directory that will contain the report
    os.makedirs(args.output_path, exist_ok=True)

    # Load from the text file only the columns containing the output of the four detectors
    log.info('Loading metadata for test "{0}"'.format(args.input_url))
    metadata = load_metadata(args.input_url)
    params = build_dict_from_results(pol_name=args.polarimeter_name,
                                     metadata=metadata)
    params['data_url'] = args.input_url

    save_parameters_to_json(params=dict(params, **get_code_version_params()),
                            output_file_name=os.path.join(args.output_path,
                                                          'bias_configuration.json'))

    create_report(params=params,
                  md_template_file='biases.md',
                  md_report_file='bias_report.md',
                  html_report_file='bias_report.html',
                  output_path=args.output_path)


if __name__ == '__main__':
    main()
