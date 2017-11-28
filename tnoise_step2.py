#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
import logging as log
import os
from shutil import copyfile
import sys
from typing import Any, Dict

import emcee
import simplejson as json
import numpy as np
import scipy
import yaml
from json_save import save_parameters_to_json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from corner import corner

from file_access import load_timestream
from reports import create_report

NUM_OF_PARAMS = 3
PARAMETER_NAMES = ('average_gain', 'gain_prod', 'tnoise')
Parameters = namedtuple('Parameters', PARAMETER_NAMES)

# These are default values, the user can update them using command-line switches
DEFAULT_WALKERS = 100
DEFAULT_BURN_IN = 3000
DEFAULT_ITERATIONS = 100000
TEST_DB_PATH = os.path.join(os.path.dirname(__file__),
                            'polarimeter_info.yaml')

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
    '''Base class for likelihood computation

    This class is used as a container for the arrays to be used in the search
    for the best estimate of the gain and noise temperature. It is never used
    directly, only as a base class for two derived classes
    ("LogLikelihoodForFit" and "LogLikelihoodForMCMC").'''

    def __init__(self, blind_channel, voltages, voltage_std,
                 temperatures_a, temperatures_b):
        assert len(voltages) == 4, '''
            "voltages" must be an array of 4 elements, one for each PWR output'''

        assert blind_channel in (
            0, 3), 'the blind channel must be either PWR0 or PWR3'
        self.blind_channel = blind_channel

        self.voltages = voltages
        self.voltage_std = voltage_std
        self.temperatures_a = temperatures_a
        self.temperatures_b = temperatures_b

        # Each pair contains the sign of the variable term and the value of
        # cosΔφ/sinΔφ in the analytical expression for the four outputs PWR0-3
        if self.blind_channel == 0:
            self.pwr_constants = [
                (+1.0, 1.0),
                (+1.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 1.0),
            ]
        else:
            # Swap PWR0 (Q1) and PWR3 (Q2) with respect to the other case
            self.pwr_constants = [
                (-1.0, 1.0),
                (+1.0, 0.0),
                (-1.0, 0.0),
                (+1.0, 1.0),
            ]


class LogLikelihoodForMCMC(LogLikelihood):
    '''Callable class to compute the likelihood for a MCMC analysis

    This class implements the calculation of the likelihood and the priors
    needed to find the optimal estimate for the gain and noise temperature
    through a MCMC analysis.
    '''

    def __init__(self, blind_channel, voltages, voltage_std,
                 temperatures_a, temperatures_b):
        super().__init__(blind_channel, voltages, voltage_std,
                         temperatures_a, temperatures_b)

    @staticmethod
    def lnprior(theta):
        # Destructure the parameters
        params = Parameters(*theta)

        if params.average_gain < 0.0 or params.gain_prod < 0.0 or params.tnoise <= 0.0:
            return -np.inf

        return 0.0

    def __call__(self, theta):
        result = self.lnprior(theta)
        if not np.isfinite(result):
            return -np.inf

        # Destructure the parameters
        params = Parameters(*theta)

        # This term is the same for all the four PWR outputs
        fixed_term = (
            params.average_gain *
            (self.temperatures_a + self.temperatures_b + 2.0 * params.tnoise)
        )

        diff_term = params.gain_prod * \
            (self.temperatures_a - self.temperatures_b)

        # Cycle over the four PWR outputs
        for set_idx, set_params in enumerate(self.pwr_constants):
            sign, phi = set_params
            estimates = -0.25 * (
                fixed_term +
                sign * phi * diff_term
            )
            std = self.voltage_std[set_idx]

            result += -0.5 * \
                np.sum(((self.voltages[set_idx] - estimates) / std) ** 2 +
                       np.log(2.0 * np.pi * std**2))

        return result


class LogLikelihoodForFit(LogLikelihood):
    '''Callable class to compute the likelihood for a MCMC analysis

    This class implements the calculation of the likelihood needed to find the
    optimal estimate for the gain and noise temperature by direct minimization.
    '''

    def __init__(self, blind_channel, voltages, voltage_std,
                 temperatures_a, temperatures_b):
        super().__init__(blind_channel, voltages, voltage_std,
                         temperatures_a, temperatures_b)

    def __call__(self, xdata, average_gain, prod_gain, tnoise):
        del xdata  # We're not going to use it

        # This term is the same for all the four PWR outputs
        fixed_term = (
            average_gain *
            (self.temperatures_a + self.temperatures_b + 2.0 * tnoise)
        )

        diff_term = prod_gain * (self.temperatures_a - self.temperatures_b)

        # Cycle over the four PWR outputs
        result = np.array([])
        for set_params in self.pwr_constants:
            sign, trig = set_params
            estimates = -0.25 * (fixed_term + sign * trig * diff_term)

            result = np.concatenate((result, estimates))

        return result


def extract_polarimeter_params(test_db: Dict[str, Any],
                               polarimeter_name: str) -> Dict[str, Any]:
    '''Return information about a polarimeter

    Search for tests related to the polarimeter in the test database (this is
    usually loaded from a YAML file). Return the part of the database that
    contains details about the test for this polarimeter.
    '''
    if len(test_db[polarimeter_name]) != 1:
        log.fatal('I expected just one test for polarimeter %s, but I found %d of them',
                  polarimeter_name, len(test_db[polarimeter_name]))

    pol_params = test_db[polarimeter_name][0]
    if pol_params['defective']:
        log.fatal('Polarimeter %s seems to be defective',
                  polarimeter_name)

    return pol_params


def extract_average_values(power_data, metadata, tnoise1_results, num):
    '''Compute statistics of PWR output for each temperature step

    Tries to find `num` temperature steps by cycling over the statistics of the
    three non-blind PWR outputs produced by the tnoise1 analysis step.

    Return a pair (VOLT, STD). Both VOLT and STD are 4-element lists of NumPy
    arrays (one per each PWR output), each containing the voltages (VOLT) and
    standard deviations (STD) for each temperature step.'''

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
    offsets = [metadata['detector_outputs'][-1]['{0}_adu'.format(det)]
               for det in ('q1', 'u1', 'u2', 'q2')]

    voltages = [np.empty(len(regions)) for i in range(4)]
    voltage_std = [np.empty(len(regions)) for i in range(4)]
    for idx, cur_region in enumerate(regions):
        start, stop = [cur_region[x] for x in ('index0', 'index1')]

        for i in range(4):
            arr = power_data[start:stop, i]
            voltages[i][idx] = np.mean(arr) - offsets[i]
            voltage_std[i][idx] = np.std(arr)

    return voltages, voltage_std


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


def assemble_results(polarimeter_name: str, log_ln: LogLikelihood, popt, pcov):
    '''Build a dictionary containing all the relevant results of the analysis

    The dictionary is meant to be saved in a JSON file, and used as input to
    produce the Markdown/HTML report.'''

    result = {
        'polarimeter_name': polarimeter_name,
        'title': ('Noise temperature analysis for polarimeter {0}'
                  .format(polarimeter_name)),
        'mcmc': False,
        'steps': [{
            't_load_a_K': log_ln.temperatures_a[idx],
            't_load_b_K': log_ln.temperatures_b[idx],
            'pwr0_adu': log_ln.voltages[0][idx],
            'pwr0_rms_adu': log_ln.voltage_std[0][idx],
            'pwr1_adu': log_ln.voltages[1][idx],
            'pwr1_rms_adu': log_ln.voltage_std[1][idx],
            'pwr2_adu': log_ln.voltages[2][idx],
            'pwr2_rms_adu': log_ln.voltage_std[2][idx],
            'pwr3_adu': log_ln.voltages[3][idx],
            'pwr3_rms_adu': log_ln.voltage_std[3][idx],
        } for idx in range(len(log_ln.temperatures_a))],
    }

    # Save the average and RMS of the fitting parameters
    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        result[param_name] = {
            'mean': popt[param_idx],
            'std': np.sqrt(pcov[param_idx, param_idx])
        }

    # Since NumPy matrices cannot be saved in JSON files, we convert it into a
    # straight list of floats
    result['covariance_matrix'] = [float(x) for x in pcov.flatten()]

    return result


def estimate_statistics(arr) -> Dict['str', float]:
    'Small helper function to compute some statistics for an array'

    mean = np.mean(arr)
    std = np.std(arr)

    lower, median, upper = np.percentile(arr, (2.5, 50, 97.5))

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'lower_err_95CL': median - lower,
        'upper_err_95CL': upper - median,
    }


def assemble_results_for_MCMC(sampler, burn_in: int):
    'Work like "assemble_results", but the information is related to a MCMC analysis'

    # Perform an autocorrelation analysis to properly pick the samples from the MC
    try:
        autocorr_lengths = []
        autocorr_length = int(np.max(sampler.get_autocorr_time()) * 2)
        log.info('the autocorrelation time of the series is %.1f',
                 autocorr_length)
    except emcee.autocorr.AutocorrError as exc:
        log.warning('unable to find a good autocorrelation time, '
                    'setting it to one (%s)', exc)
        autocorr_length = 1

    series = sampler.chain[:, burn_in::autocorr_length,
                           :].reshape(-1, NUM_OF_PARAMS)

    result = {
        'num_of_walkers': sampler.chain.shape[0],
        'num_of_iterations': sampler.chain.shape[1],
        'burn_in_length': burn_in,
        'autocorrelation_length': autocorr_length,
        'num_of_samples': series.shape[0],
        'mcmc': True,
    }

    for param_idx, param_name in enumerate(PARAMETER_NAMES):
        result['mcmc_' + param_name] = \
            estimate_statistics(series[:, param_idx])

    return result


def create_plots(log_ln, params, output_path: str):
    fig = plt.figure()
    temperatures_a = np.array([x['t_load_a_K'] for x in params['steps']])
    temperatures_b = np.array([x['t_load_b_K'] for x in params['steps']])

    pol_gain, gain_prod = \
        [params[x]['mean'] for x in ('average_gain', 'gain_prod')]

    min_volt, max_volt = None, None
    for pwr_idx, pwr_params in enumerate(log_ln.pwr_constants):
        sign, trig = pwr_params
        estimates = -0.25 * (
            pol_gain * (temperatures_a + temperatures_b +
                        2.0 * params['tnoise']['mean']) +
            sign * trig * gain_prod * (temperatures_a - temperatures_b)
        )

        label = 'pwr{0}_adu'.format(pwr_idx)
        voltages = np.array([x[label] for x in params['steps']])
        if pwr_idx == 0:
            min_volt, max_volt = np.min(voltages), np.max(voltages)
        else:
            min_volt = min(min_volt, np.min(voltages))
            max_volt = max(max_volt, np.max(voltages))

        label = 'pwr{0}_rms_adu'.format(pwr_idx)
        voltage_std = np.array([x[label] for x in params['steps']])

        plt.scatter(voltages, estimates, label='PWR{0}'.format(pwr_idx))

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

    lincorr_plot_file_path = os.path.join(
        output_path, 'tnoise_linear_correlation.svg')
    plt.savefig(lincorr_plot_file_path, bbox_inches='tight')
    log.info('linear correlation plot saved to file "%s"',
             lincorr_plot_file_path)


def create_MCMC_plots(log_ln, params, chain, output_path: str):
    # Corner plot
    fig = plt.figure()
    samples = chain[:,
                    params['burn_in_length']::params['autocorrelation_length'],
                    :].reshape(-1, NUM_OF_PARAMS)
    fig = plt.figure()
    corner(samples, bins=50, labels=('$\\frac{1}{2}(g_a^2 + g_b^2)$',
                                     '$g_a \\times g_b$',
                                     '$T_n$'))
    corner_plot_file_path = os.path.join(output_path, 'tnoise_corner_plot.svg')
    plt.savefig(corner_plot_file_path, bbox_inches='tight')
    log.info('corner plot saved to file "%s"', corner_plot_file_path)


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``raw_file``
    - ``tnoise1_results``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--mcmc', action='store_true',
                        help='''Run a MCMC analysis on the data as well
                        (usually not needed)''')
    parser.add_argument('--num-of-walkers', dest='walkers',
                        type=int, default=DEFAULT_WALKERS,
                        help='''Number of walkers to use in the simulation
                        (default: {0})'''.format(DEFAULT_WALKERS))
    parser.add_argument('--num-of-iterations', dest='iterations',
                        type=int, default=DEFAULT_ITERATIONS,
                        help='''Number of walkers to use in the simulation
                        (default: {0})'''.format(DEFAULT_ITERATIONS))
    parser.add_argument('--burn-in', type=int, default=DEFAULT_BURN_IN,
                        help='''Number of samples at the beginning to exclude
                        (default: {0})'''.format(DEFAULT_ITERATIONS))
    parser.add_argument('--test-db-path', default=TEST_DB_PATH, type=str,
                        help='''Path to the test database YAML file
                        (default is "{0}")'''.format(TEST_DB_PATH))
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

    log.info('reading file "%s"', args.test_db_path)
    with open(args.test_db_path, 'rt') as yaml_file:
        test_db = yaml.load(yaml_file)

    log.info('reading file "%s"', args.raw_file)
    metadata, data = load_timestream(args.raw_file)

    if not args.polarimeter_name in test_db:
        log.fatal('polarimeter "%s" not present in the test database',
                  args.polarimeter_name)

    temperatures_a, temperatures_b = extract_temperatures(metadata)
    log.info('temperatures for load A: %s',
             str(temperatures_a))
    log.info('temperatures for load B: %s',
             str(temperatures_b))

    voltages, voltage_std = extract_average_values(data.power, metadata, tnoise1_results,
                                                   num=len(temperatures_a))
    for idx, arr in enumerate(voltages):
        log.info('voltages for PWR%d: %s',
                 idx, ', '.join(['{0:.1f}'.format(x) for x in arr]))
        log.info('voltage RMS for PWR%d: %s',
                 idx, ', '.join(['{0:.1f}'.format(x) for x in voltage_std[idx]]))

    log_ln = LogLikelihoodForFit(blind_channel=tnoise1_results['blind_channel'],
                                 voltages=voltages,
                                 voltage_std=voltage_std,
                                 temperatures_a=temperatures_a,
                                 temperatures_b=temperatures_b)

    popt, pcov = scipy.optimize.curve_fit(
        log_ln, None, np.array(voltages).flatten(),
        p0=np.array([1e4, 1e4, 30.0]),
        sigma=np.array(voltage_std).flatten())
    log.info('results of the fit: %s',
             ', '.join(['{0} = {1:.2f} ± {2:.2f}'
                        .format(n, popt[i], np.sqrt(pcov[i, i]))
                        for i, n in enumerate(PARAMETER_NAMES)]))

    params = assemble_results(args.polarimeter_name,
                              log_ln, popt, pcov)

    if args.mcmc:
        log.info('running a MCMC analysis with %d walkers for %d steps',
                 args.walkers, args.iterations)

        # Overwrite log_ln (which was of type LogLikelihoodForFit) with a new
        # object
        log_ln = LogLikelihoodForMCMC(blind_channel=tnoise1_results['blind_channel'],
                                      voltages=voltages,
                                      voltage_std=voltage_std,
                                      temperatures_a=temperatures_a,
                                      temperatures_b=temperatures_b)

        sampler = emcee.EnsembleSampler(nwalkers=args.walkers,
                                        dim=NUM_OF_PARAMS,
                                        lnpostfn=log_ln)

        # Reuse the results of the fitting to initialize the starting points of
        # the MCMC walkers
        starting_point = emcee.utils.sample_ball(popt,
                                                 std=np.sqrt(np.diag(pcov)),
                                                 size=args.walkers)
        for idx, _ in enumerate(sampler.sample(starting_point,
                                               iterations=args.iterations)):
            if idx % 1000 == 0 and idx > 0:
                log.info('Running step %d/%d (%.1f%%)',
                         idx, args.iterations,
                         (idx * 100.0) / args.iterations)

        # Save the chain of samples: this is *pure gold* for debugging!
        chain_file_path = os.path.join(args.output_path,
                                       'tnoise_mcmc_chain.npy')
        np.save(chain_file_path, sampler.chain)
        log.info('chain of MCMC samples saved in file "%s"',
                 chain_file_path)

        log.info('analyzing the MCMC data, burn-in length is %d',
                 args.burn_in)

        # Merge the "params" dictionary with the return value of
        # "assemble_results_for_MCMCM"
        params = {
            **params,
            **assemble_results_for_MCMC(sampler, args.burn_in)
        }

        create_MCMC_plots(log_ln, params, sampler.chain, args.output_path)

    save_parameters_to_json(params=params,
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
