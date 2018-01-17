#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from argparse import ArgumentParser
from collections import namedtuple
import logging as log
import os
import sys
from typing import Any, Dict, Union

import simplejson as json
import numpy as np
import scipy.optimize as opt
from json_save import save_parameters_to_json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.patheffects as PathEffects

from file_access import load_timestream
from reports import create_report, get_code_version_params

SAMPLING_FREQUENCY_HZ = 25.0
# Any temperature variation below this will be considered zero
MIN_TEMPERATURE_STEP_K = 2.5

NUM_OF_PARAMS = 3
PARAMETER_NAMES = ('gain_q1', 'gain_u1', 'gain_u2',
                   'gain_q2', 'unbalance', 'tnoise')
Parameters = namedtuple('Parameters', PARAMETER_NAMES)

# List of housekeeping temperatures used to estimate the temperature of the two
# loads. They are usually load from the test database
Housekeepings = namedtuple('Housekeepings', [
    't_load_a_1',
    't_load_a_2',
    't_load_b_1',
    't_load_b_2',
    't_cross_guide_1',
    't_cross_guide_2',
    't_polarimeter_1',
    't_polarimeter_2',
])

# List of temperatures computed using Claudio Pincella's equations (see the
# function "compute_T_load_attenuations")
ChamberTemperatures = namedtuple('ChamberTemperatures', [
    't_average_cross_guide',
    't_magic_t',
    't_load_a',
    't_ambient',
    't_bright_from_cross_guide',
    't_bright_port_a',
    't_bright_from_b',
])


def detector_name(pwr_idx):
    'Given a number 0..3, return a string containing the name of the polarimetric detector'
    if pwr_idx == 0:
        return 'Q1 (PWR0)'
    elif pwr_idx == 1:
        return 'U1 (PWR1)'
    elif pwr_idx == 2:
        return 'U2 (PWR2)'
    elif pwr_idx == 3:
        return 'Q2 (PWR3)'
    else:
        raise ValueError('invalid channel number {0}'.format(pwr_idx))


def compute_T_load_attenuations(temperatures: Housekeepings,
                                t_ambient: float = 300.0,
                                wg_att: float = 0.110,
                                tot_att: float = 0.993) -> ChamberTemperatures:
    '''
    This function calculates the effective brightness temperatures of the loads
    A and B, considering the attenuations due to the experimental setup. The
    attenuations have been measured through dedicated test. For more details see
    Annex D of Claudio Pincella's master thesis.

    Parameters
    ----------
    t_ambient    : float,
                   Ambient temperature, usually 300 K
    temperatures : Housekeepings
                   Temperatures of the experimental setup components as read on the
                   display of the temperature control panel
    wg_att       : float,
                   The linear attenuation due to the inox wave guide
    tot_att      : float,
                   The linear attenuation due to the combination of RF guide,
                   cross guide, and wave guide.

    Returns
    -------
    out : ChamberTemperatures
    '''
    t_a1, t_a2, t_b1, t_b2, t_c1, t_c2, t_p1, t_p2 = temperatures
    t_avg_cross_guide = (t_c1 + t_c2) * 0.5
    t_avg_magic_t_polarimeter = (t_p1 + t_p2) * 0.5
    brightness_t_load_a = t_a1 * (1 - wg_att) + wg_att * (t_a2 + t_c2) * 0.5
    brightness_t_from_ambient = ((t_ambient * (1 - wg_att) +
                                  wg_att * (t_ambient + t_avg_cross_guide) * 0.5) *
                                 (1 - tot_att))
    brightness_t_from_cross_guide = t_avg_cross_guide * (1 - tot_att)
    brightness_t_port_a_total = brightness_t_load_a + \
        brightness_t_from_ambient + brightness_t_from_cross_guide
    brightness_t_from_load_b = t_b1 * (1 - wg_att) + wg_att * \
        (t_b1 + t_avg_magic_t_polarimeter) * 0.5

    return ChamberTemperatures(
        t_average_cross_guide=t_avg_cross_guide,
        t_magic_t=t_avg_magic_t_polarimeter,
        t_load_a=brightness_t_load_a,
        t_ambient=brightness_t_from_ambient,
        t_bright_from_cross_guide=brightness_t_from_cross_guide,
        t_bright_port_a=brightness_t_port_a_total,
        t_bright_from_b=brightness_t_from_load_b
    )


class LogLikelihood:
    '''Class for likelihood computation

    This class is used as a container for the arrays to be used in the search
    for the best estimate of the gain and noise temperature. It can be passed
    as a parameter to scipy.optimize.curve_fit.'''

    def __init__(self, voltages, voltage_std, wn_level, wn_error, nsamples,
                 temperatures_a, temperatures_b, phsw_state):
        assert len(voltages) == 4, '''
            "voltages" must be an array of 4 elements, one for each PWR output'''

        self.voltages = voltages
        self.voltage_std = voltage_std
        self.wn_level = wn_level
        self.wn_error = wn_error
        self.nsamples = nsamples
        self.temperatures_a = temperatures_a
        self.temperatures_b = temperatures_b
        self.phsw_state = phsw_state

        # Each pair contains the weights applied to the following terms:
        # 1. T_a
        # 2. T_b
        # 3. (T_a + T_b) / 2 * unbalance
        # in the analytical expression for the four outputs PWR0-3
        if self.phsw_state in ('0101', '1010'):
            self.ta_tb_weights = [
                (1.0, 0.0, 1.0),
                (0.5, 0.5, 0.0),
                (0.5, 0.5, 0.0),
                (0.0, 1.0, 1.0),
            ]
        elif self.phsw_state in ('0110', '1001'):
            # Swap PWR0 (Q1) and PWR3 (Q2) with respect to the other case
            self.ta_tb_weights = [
                (0.0, 1.0, 1.0),
                (0.5, 0.5, 0.0),
                (0.5, 0.5, 0.0),
                (1.0, 0.0, 1.0),
            ]

    def load_temp_at_detector(self, pwr_idx, unbalance):
        ta_weight, tb_weight, cross_weight = self.ta_tb_weights[pwr_idx]
        return (
            ta_weight * self.temperatures_a +
            tb_weight * self.temperatures_b +
            cross_weight * (self.temperatures_a +
                            self.temperatures_b) / 2 * unbalance
        )

    def __call__(self, xdata, gain_q1, gain_u1, gain_u2, gain_q2, unbalance, tnoise):
        del xdata  # We're not going to use it

        result = np.array([])
        for pwr_idx, gain in enumerate((gain_q1, gain_u1, gain_u2, gain_q2)):
            temperature = self.load_temp_at_detector(pwr_idx, unbalance)
            estimates = -gain * (temperature + tnoise)
            result = np.concatenate((result, estimates))

        return result


def calc_wn_level(values):
    from scipy.signal import welch

    _, psd = welch(values, fs=SAMPLING_FREQUENCY_HZ,
                   window='hann', nperseg=2.0 * SAMPLING_FREQUENCY_HZ,
                   detrend='linear')

    wn_level = np.mean(psd)
    err = np.std(psd) / np.sqrt(len(psd))
    return float(wn_level), float(err)


def extract_average_values(power_data, dem_data, metadata, tnoise1_results, num, offsets=None):
    '''Compute statistics of PWR output for each temperature step

    Tries to find `num` temperature steps by cycling over the statistics of the
    three non-blind PWR outputs produced by the tnoise1 analysis step.

    Return a pair (VOLT, STD, WN, WN_ERR, SAMP). All the elements of the tuple
    but SAMP are 4-element lists of NumPy arrays (one per each PWR output), each
    containing the voltages (VOLT), standard deviations (STD), white noise
    levels (WN) and white noise error (WN_ERR) for each temperature step. SAMP
    is a 1-D NumPy array containing the number of samples used in each of the
    four PWR outputs.'''

    # ID of the PWR output which will be used as "reference" (i.e., we are
    # going to use the regions detected using this PWR output)
    regions = None
    region_lengths = set()
    for key, val in tnoise1_results['regions'].items():
        if len(val) == num:
            regions = val
            break
        region_lengths.add(len(val))

    if not regions:
        log.fatal('unable to find {0} temperature steps in the data, only {1} were available'
                  .format(num, ', '.join([str(x) for x in region_lengths])))
        sys.exit(1)

    # These are the offsets acquired when the HEMTs were turned off
    if offsets is None:
        offsets = [metadata['detector_outputs'][-1]['{0}_adu'.format(det)]
                   for det in ('q1', 'u1', 'u2', 'q2')]

    voltages = [np.empty(len(regions)) for i in range(4)]
    voltage_std = [np.empty(len(regions)) for i in range(4)]
    wn_level = [np.empty(len(regions)) for i in range(4)]
    wn_err = [np.empty(len(regions)) for i in range(4)]
    nsamples = np.empty(len(regions))
    for idx, cur_region in enumerate(regions):
        start, stop = [cur_region[x] for x in ('index0', 'index1')]
        nsamples[idx] = stop - start + 1

        for i in range(4):
            arr = power_data[start:stop, i]
            voltages[i][idx] = np.mean(arr) - offsets[i]
            voltage_std[i][idx] = np.std(arr)
            wn_level[i][idx], wn_err[i][idx] = calc_wn_level(
                dem_data[start:stop, i])

    return voltages, voltage_std, wn_level, wn_err, nsamples, offsets


def extract_temperatures(test_metadata):
    '''Return the temperatures of the two loads as seen by the polarimeter

    This function loads the housekeeping temperatures from the test database for
    the relevant polarimeter, and then uses "compute_T_load_attenuations" to
    determine the temperature of loads A and B at each temperature step.

    Return a pair (TEMP_A, TEMP_B), where both TEMP_A and TEMP_B are NumPy
    arrays.'''

    steps = test_metadata['temperatures']
    temperatures_a = np.empty(len(steps))
    temperatures_b = np.empty(len(steps))
    for idx, cur_step in enumerate(steps):
        hk = Housekeepings(
            t_load_a_1=cur_step['t_load_a_1_K'],
            t_load_a_2=cur_step['t_load_a_2_K'],
            t_load_b_1=cur_step['t_load_b_1_K'],
            t_load_b_2=cur_step['t_load_b_2_K'],
            t_cross_guide_1=cur_step['t_cross_guide_1_K'],
            t_cross_guide_2=cur_step['t_cross_guide_2_K'],
            t_polarimeter_1=cur_step['t_polarimeter_1_K'],
            t_polarimeter_2=cur_step['t_polarimeter_2_K'],
        )
        t_estimates = compute_T_load_attenuations(hk)
        temperatures_a[idx] = t_estimates.t_bright_port_a
        temperatures_b[idx] = t_estimates.t_bright_from_b

    return temperatures_a, temperatures_b


def y_factor_pairs(num_of_steps):
    '''Return a pair of indexes representing two temperature steps for the Y-factor

    Each pair contains the (i, j) indexes of the i-th and j-th temperature
    step to use in a Y-factor computation.'''

    from itertools import product
    return [x for x in product(range(num_of_steps), range(num_of_steps))
            if x[0] < x[1]]


def estimate_tnoise_and_gain(temp_a, temp_b, out_a, out_b):
    'Compute the noise temperature and gain from two temperature steps'

    if np.abs(temp_a - temp_b) < MIN_TEMPERATURE_STEP_K:
        return np.nan, np.nan

    # The equation of the line is given by: y == out_a + gain * (x - temp_a)
    gain = (out_a - out_b) / (temp_a - temp_b)
    noise_temp = out_a / gain - temp_a

    return noise_temp, gain


def y_factor_estimates(log_ln, unbalance):
    result = []

    for idx1, idx2 in y_factor_pairs(len(log_ln.temperatures_a)):
        for pwr_idx in (0, 1, 2, 3):
            temperatures = log_ln.load_temp_at_detector(
                pwr_idx=pwr_idx, unbalance=unbalance)
            tnoise, gain = estimate_tnoise_and_gain(temp_a=temperatures[idx1],
                                                    temp_b=temperatures[idx2],
                                                    out_a=log_ln.voltages[pwr_idx][idx1],
                                                    out_b=log_ln.voltages[pwr_idx][idx2])
            if np.isfinite(tnoise) and np.isfinite(gain):
                result.append({
                    'detector_idx': pwr_idx,
                    'detector_name': detector_name(pwr_idx),
                    'step_1_idx': idx1,
                    'step_2_idx': idx2,
                    'temperature_1': temperatures[idx1],
                    'temperature_2': temperatures[idx2],
                    'output_1': log_ln.voltages[pwr_idx][idx1],
                    'output_2': log_ln.voltages[pwr_idx][idx2],
                    'tnoise': tnoise,
                    'gain': gain,
                })

    result.sort(key=lambda x: (x['detector_idx'],
                               x['step_1_idx'], x['step_2_idx']))

    return result


def assemble_results(polarimeter_name: str, log_ln: LogLikelihood, popt, pcov):
    '''Build a dictionary containing all the relevant results of the analysis

    The dictionary is meant to be saved in a JSON file, and used as input to
    produce the Markdown/HTML report.'''

    result = {
        'polarimeter_name': polarimeter_name,
        'title': ('Noise temperature analysis for polarimeter {0}'
                  .format(polarimeter_name)),
        'analysis_method': 'non-linear fit',
        'phsw_state': log_ln.phsw_state,
        'parameters': PARAMETER_NAMES,
        'steps': [{
            't_load_a_K': log_ln.temperatures_a[idx],
            't_load_b_K': log_ln.temperatures_b[idx],
            'nsamples': log_ln.nsamples[idx],
            'pwr0_adu': log_ln.voltages[0][idx],
            'pwr0_rms_adu': log_ln.voltage_std[0][idx],
            'pwr0_wn_level': log_ln.wn_level[0][idx],
            'pwr0_wn_error': log_ln.wn_error[0][idx],
            'pwr1_adu': log_ln.voltages[1][idx],
            'pwr1_rms_adu': log_ln.voltage_std[1][idx],
            'pwr1_wn_level': log_ln.wn_level[1][idx],
            'pwr1_wn_error': log_ln.wn_error[1][idx],
            'pwr2_adu': log_ln.voltages[2][idx],
            'pwr2_rms_adu': log_ln.voltage_std[2][idx],
            'pwr2_wn_level': log_ln.wn_level[2][idx],
            'pwr2_wn_error': log_ln.wn_error[2][idx],
            'pwr3_adu': log_ln.voltages[3][idx],
            'pwr3_rms_adu': log_ln.voltage_std[3][idx],
            'pwr3_wn_level': log_ln.wn_level[3][idx],
            'pwr3_wn_error': log_ln.wn_error[3][idx],
        } for idx in range(len(log_ln.temperatures_a))],
    }

    # Save the average and RMS of the fitting parameters
    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        result[param_name] = {
            'mean': popt[param_idx],
            'std': np.sqrt(pcov[param_idx, param_idx])
        }

    result['y_factor_estimates'] = y_factor_estimates(log_ln, popt.unbalance)

    # Since NumPy matrices cannot be saved in JSON files, we convert it into a
    # straight list of floats
    result['covariance_matrix'] = [float(x) for x in pcov.flatten()]

    return result


def save_plot(output_dir, file_name):
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    log.info('plot saved to file "%s"', file_path)


def create_timestream_plot(log_ln, params, output_path):
    _ = plt.figure(figsize=(8, 8))

    temperatures_a = np.array([x['t_load_a_K'] for x in params['steps']])
    temperatures_b = np.array([x['t_load_b_K'] for x in params['steps']])

    best_fit = Parameters(*[params[x]['mean'] for x in PARAMETER_NAMES])
    model_estimates = log_ln(None, *best_fit).reshape(4, -1)

    if np.var(temperatures_a) > np.var(temperatures_b):
        varying_t = temperatures_a
        load_name = 'A'
    else:
        varying_t = temperatures_b
        load_name = 'B'

    plt.subplot(211)
    pwr = []
    for pwr_idx in (0, 1, 2, 3):
        cur_pwr = np.array([log_ln.voltages[pwr_idx][i]
                            for i in range(len(log_ln.temperatures_a))])
        base_line, = plt.plot(varying_t, cur_pwr, '-o',
                              label=detector_name(pwr_idx))
        plt.plot(varying_t, model_estimates[pwr_idx],
                 alpha=0.5, color=base_line.get_color())
        pwr.append(cur_pwr)

    plt.legend()
    plt.ylabel('Output [ADU]')

    plt.subplot(212)
    for pwr_idx, gain in enumerate((best_fit.gain_q1,
                                    best_fit.gain_u1,
                                    best_fit.gain_u2,
                                    best_fit.gain_q2)):
        plt.plot(varying_t, (pwr[pwr_idx] - model_estimates[pwr_idx]) / gain,
                 '-o', linestyle='--')

    plt.ylabel('Measure \u2212 model [K]')
    plt.xlabel('$T_{0}$ [K]'.format(load_name))
    save_plot(output_path, 'temperature_timestream.svg')


def create_model_match_plot(log_ln, params, output_path):
    fig = plt.figure()

    best_fit = [params[x]['mean'] for x in PARAMETER_NAMES]

    # These two variables are used to draw the y=x blue line
    min_volt, max_volt = None, None
    model_estimates = log_ln(None, *best_fit).reshape(4, -1)
    for pwr_idx in (0, 1, 2, 3):
        label = 'pwr{0}_adu'.format(pwr_idx)
        voltages = np.array([x[label] for x in params['steps']])

        # Update min_volt and max_volt
        if pwr_idx == 0:
            min_volt, max_volt = np.min(voltages), np.max(voltages)
        else:
            min_volt = min(min_volt, np.min(voltages))
            max_volt = max(max_volt, np.max(voltages))

        plt.scatter(
            voltages, model_estimates[pwr_idx], label=detector_name(pwr_idx))

    # Straight line representing the 1:1 case (perfect match between
    # measurements and model)
    plt.plot([min_volt, max_volt], [min_volt, max_volt],
             color='blue', label='1:1 relationship')

    plt.legend()
    plt.xlabel('Measured output [ADU]')
    plt.ylabel('Model estimate [ADU]')

    # Use a 1:1 ratio for the length of the two axes of the plot (it's more
    # visually pleasant)
    plt.axes().set_aspect('equal')

    save_plot(output_path, 'tnoise_linear_correlation.svg')


def create_tnoise_plot(log_ln, params, output_path):
    _ = plt.figure(figsize=(8, 5))

    best_fit = Parameters(*[params[x]['mean'] for x in PARAMETER_NAMES])

    for pwr_idx in (0, 1, 2, 3):
        temperature = log_ln.load_temp_at_detector(
            pwr_idx=pwr_idx, unbalance=best_fit.unbalance)
        plt.scatter(
            temperature, log_ln.voltages[pwr_idx], label=detector_name(pwr_idx), zorder=3)

    pwr_colors = {
        0: '#202020',
        1: '#606060',
        2: '#909090',
        3: '#b0b0b0',
    }
    for y_estimate in params['y_factor_estimates']:
        tnoise, gain, temp1, temp2, out1 = [y_estimate[x] for x in (
            'tnoise', 'gain', 'temperature_1', 'temperature_2', 'output_1')]
        if np.abs(temp2 - temp1) < MIN_TEMPERATURE_STEP_K:
            continue

        x = np.linspace(-tnoise, np.max([temp1, temp2]), 2)
        pwr_col = pwr_colors[y_estimate['detector_idx']]
        plt.plot(x, out1 + gain * (x - temp1), color=pwr_col, zorder=2)
        plt.scatter(-tnoise, 0.0, color=pwr_col, zorder=1)

    plt.xlabel('Temperature of the load [K]')
    plt.ylabel('Output [ADU]')
    plt.legend()

    save_plot(output_path, 'tnoise_estimates_from_y_factor.svg')


def create_tnoise_matrix_plot(log_ln, params, output_path):

    y_factor_estimates = params['y_factor_estimates']
    if not y_factor_estimates:
        log.warning('no Y-factor estimates')
        return

    min_tnoise = np.min([x['tnoise'] for x in y_factor_estimates])
    max_tnoise = np.max([x['tnoise'] for x in y_factor_estimates])
    detectors = set([x['detector_name'] for x in y_factor_estimates])

    fig = plt.figure(figsize=(10, 4))

    for cur_idx, cur_det in enumerate(sorted(detectors)):
        ax = plt.subplot(1, len(detectors), 1 + cur_idx)

        cur_estimates = [x for x in y_factor_estimates
                         if x['detector_name'] == cur_det]
        matr_size = np.max(np.matrix(
            [[x['step_1_idx'], x['step_2_idx']] for x in cur_estimates]).flatten()) + 1

        # Build the strings used to label the ticks along the two axes
        step_label = []
        for i in range(matr_size):
            # Find some temperature whose index matches "i"
            bunch = [x['temperature_1'] for x in cur_estimates if x['step_1_idx'] == i] \
                + [x['temperature_2']
                    for x in cur_estimates if x['step_2_idx'] == i]
            if bunch:
                step_label.append('{0:.1f} K'.format(bunch[0]))

        tnoise_matr = np.zeros((matr_size, matr_size)) * np.nan
        for est in cur_estimates:
            tnoise_matr[est['step_1_idx'], est['step_2_idx']] = est['tnoise']

        im = ax.matshow(np.ma.masked_invalid(tnoise_matr),
                        cmap=plt.cm.plasma,
                        vmin=min_tnoise, vmax=max_tnoise)

        # Text labels over each square in the grid
        for (i, j), z in np.ndenumerate(tnoise_matr):
            if z > 0.0:
                ax.text(j, i, '${:0.1f}\,K$'.format(z),
                        color="w",
                        ha='center', va='center',
                        path_effects=[PathEffects.withStroke(linewidth=1,
                                                             foreground="b")])

        # Add an empty string at the beginning in order to skip the
        # first tick, since this is invisible (it's outside the axes)
        ax.set_xticklabels([''] + step_label)
        ax.set_yticklabels([''] + step_label)

        # Appending a '\n' is a quick hack to raise the position of the title
        ax.set_title(cur_det + '\n')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    save_plot(output_path, 'tnoise_matrix.svg')


def create_plots(*args, **kwargs):
    create_timestream_plot(*args, **kwargs)
    create_model_match_plot(*args, **kwargs)
    create_tnoise_plot(*args, **kwargs)
    create_tnoise_matrix_plot(*args, **kwargs)


def parse_offsets(s: Union[str, None]):
    if s is None:
        return None

    numbers = s.split(',')
    if len(numbers) == 4:
        try:
            return [float(x) for x in numbers]
        except ValueError:
            pass

    # If we reach this line, "s" cannot be parsed as a list of floating-point
    # numbers. So, let's assume is a file name

    with open(s, 'rt') as f:
        data = json.load(f)
        assert('detector_outputs' in data)
        return [data['detector_outputs'][x]
                for x in ('q1_adu', 'u1_adu', 'u2_adu', 'q2_adu')]


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``raw_file``
    - ``tnoise1_results``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--phsw', type=str, default=None,
                        help='State of the PHSW (e.g., "0101")')
    parser.add_argument('--offsets', type=str, default=None,
                        help='''Specify the value of the offsets. It can either
                        be a file name, in which case it will be assumed to be a
                        JSON file containing the output of the program
                        "output_average", or a comma-separated list of
                        floating-point numbers specifying the offset of the four
                        outputs PWR0, PWR1, PWR2, and PWR3. If they are not
                        specified, they will be read from the test database.''')
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter (must match the name
                        in the test database)''')
    parser.add_argument('raw_file', type=str,
                        help='''Name of the text file containing the raw
                        timelines acquired by the acquisition system''')
    parser.add_argument('tnoise1_results', type=str,
                        help='''Name of the JSON file containing the results
                        of the first step of the analysis''')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the path does not exist, it will be created''')
    return parser.parse_args()


def main():
    log.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=log.DEBUG)
    args = parse_arguments()

    log.info('reading file "%s"', args.tnoise1_results)
    with open(args.tnoise1_results, 'rt') as json_file:
        tnoise1_results = json.load(json_file)

    log.info('reading file "%s"', args.raw_file)
    metadata, data = load_timestream(args.raw_file)
    if metadata:
        phsw_state = metadata['phsw_state']
    else:
        phsw_state = args.phsw_state

    offsets = parse_offsets(args.offsets)

    temperatures_a, temperatures_b = extract_temperatures(metadata)
    log.info('temperatures for load A: %s',
             str(temperatures_a))
    log.info('temperatures for load B: %s',
             str(temperatures_b))

    voltages, voltage_std, wn_level, wn_err, nsamples, offsets = \
        extract_average_values(data.power, data.demodulated, metadata, tnoise1_results,
                               num=len(temperatures_a),
                               offsets=offsets)
    for idx, arr in enumerate(voltages):
        log.info('voltages for PWR%d: %s',
                 idx, ', '.join(['{0:.1f}'.format(x) for x in arr]))
        log.info('voltage RMS for PWR%d: %s',
                 idx, ', '.join(['{0:.1f}'.format(x) for x in voltage_std[idx]]))
        log.info('WN for PWR%d: %s',
                 idx, ', '.join(['{0:.1f}'.format(x) for x in wn_level[idx]]))

    log_ln = LogLikelihood(voltages=voltages,
                           voltage_std=voltage_std,
                           wn_level=wn_level,
                           wn_error=wn_err,
                           nsamples=nsamples,
                           temperatures_a=temperatures_a,
                           temperatures_b=temperatures_b,
                           phsw_state=phsw_state)

    popt, pcov = opt.curve_fit(
        log_ln, None, np.array(voltages).flatten(),
        p0=np.array([1e4, 1e4, 1e4, 1e4, 0.0, 30.0]),
        sigma=np.array(voltage_std).flatten())
    popt = Parameters(*popt)
    log.info('results of the fit: %s',
             ', '.join(['{0} = {1:.2f} ± {2:.2f}'
                        .format(n, popt[i], np.sqrt(pcov[i, i]))
                        for i, n in enumerate(PARAMETER_NAMES)]))

    params = assemble_results(args.polarimeter_name,
                              log_ln, popt, pcov)
    params['test_file_name'] = args.raw_file
    params['offsets'] = offsets

    save_parameters_to_json(params=dict(params, **get_code_version_params()),
                            output_file_name=os.path.join(args.output_path,
                                                          'tnoise_step2_results.json'))

    create_plots(log_ln, params, args.output_path)
    create_report(params=params,
                  md_template_file='tnoise_step2.md',
                  md_report_file='tnoise_step2_report.md',
                  html_report_file='tnoise_step2_report.html',
                  output_path=args.output_path)


if __name__ == '__main__':
    main()
