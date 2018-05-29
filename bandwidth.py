#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''Compute the bandwidth and the central frequency of the polarimeter'''


from json_save import save_parameters_to_json
from argparse import ArgumentParser
from scipy import signal
from reports import create_report, get_code_version_params
from file_access import load_timestream
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLING_FREQUENCY_HZ = 25.0  # Hz
NAMING_CONVENTION = ['PW0/Q1', 'PW1/U1', 'PW2/U2', 'PW3/Q2']


def remove_offset(nu, data, metadata):
    """
    This function computes the electronic offset and removes it from data.
    The offset is computed in the following way:
    - we compute the mean value of the data taken before switching on the RF generator: first_offset
    - we compute the mean value of the data taken after switching off the RF generator: second_offset
    (The acquisition system returns -1 if the RF generator is off.
    Otherwise it returns the frequency value)
    - we take as offset the linear fit between the points
     (min RF freq., first_offset) and (max RF freq., second_offset)
     In this way we take into account possible drifts of the electronics during the test.

    Other approaches for the offset computation are present in the code as comments.

    Parameters
    ----------
    nu      : numpy array of shape (time*sampling_rate, ),
              The frequency data.
    data    : numpy array of shape (time*sampling_rate, 4),
              The output power of the 4 detectors.
    metadata: dictionary containing the metadata of the test
              as parsed from the JSON record in the database

    Returns
    -------
    Data with electronic offset removed: 1 numpy array of shape (time*sampling_rate, 4).

    """

    # linear offset (RF generator on)
    # offset = np.zeros((len(nu), data.shape[-1]))
    # x = [38.0, 50.0]
    # for i in range(0, 4):
    # y = [data[:, i][nu == 38.0], data[:, i][nu == 50.0]]
    # line_coeff = np.polyfit(x, y, 1)
    # offset[:, i] = nu * line_coeff[0] + line_coeff[1]"""

    # offset claudio
    # firsthalf_data = data[:int(len(data)/2)]
    # firsthalf_nu = nu[:int(len(nu)/2)]
    # offset = np.mean(firsthalf_data[firsthalf_nu == -1], axis=0)

    firsthalf_data = data[:int(len(data) / 2)]
    firsthalf_nu = nu[:int(len(nu) / 2)]
    first_offset = np.median(firsthalf_data[firsthalf_nu == -1], axis=0)
    secondhalf_data = data[int(len(data) / 2):]
    secondhalf_nu = nu[int(len(nu) / 2):]

    if metadata['band'] == 'Q':
        second_offset = np.median(secondhalf_data[secondhalf_nu == -1], axis=0)
    elif metadata['band'] == 'W':
        high_nu = np.max(nu)
        second_offset = np.median(data[nu == high_nu], axis=0)
    else:
        raise ValueError('Unknown band {0} for test {1} (polarimeter STRIP{2:02d})'
                         .format(metadata['band'], metadata['id'], metadata['polarimeter_number']))

    offset = np.zeros((len(nu), data.shape[-1]))
    x = [np.min(nu[nu > 0]), np.max(nu)]
    for i in range(0, 4):
        y = [first_offset[i], second_offset[i]]
        line_coeff = np.polyfit(x, y, 1)
        offset[:, i] = nu * line_coeff[0] + line_coeff[1]

    data_nooff = data - offset
    return data_nooff


def get_frequency_range_and_data(nu, data, std_dev=True, rej=1):
    """
    This function does the following:
    - rejects data outside the frequency range imposed by the RF generator (the
    acquisition system returns -1 if the RF generator is off. Otherwise it returns the frequency value).
    - returns the mean value of the power output for each RF frequency and the associated standard
    deviation.

    Parameters
    ----------
    nu      : numpy array of shape (time*sampling_rate, ),
              The frequency data.
    data    : numpy array of shape (time*sampling_rate, 4),
              The output power of the 4 detectors.
    std_dev : boolean,
              If True (default) it will return also the standard deviation of the averaged data.
    rej     : integer,
              The number of samples rejected at each frequency step because of frequency uncertainty.

    Returns
    -------
    1 numpy array of shape (number of frequency steps, ) and 2 (or 1, if std_dev is False) numpy arrays of shape (number of frequency steps, 4).
    """
    new_nu_, new_data_ = nu[nu > 0], data[nu > 0]
    new_nu, idx, count = np.unique(
        new_nu_, return_index=True, return_counts=True)
    new_nu = np.around(new_nu, decimals=3)
    new_data = np.zeros((len(new_nu), data.shape[-1]))
    new_std_dev = np.zeros((len(new_nu), data.shape[-1]))

    for (i, j), c in zip(enumerate(idx), count):
        new_data[i] = np.median(new_data_[j + rej: j + c - rej], axis=0)
        new_std_dev[i] = np.std(
            new_data_[j + rej: j + c - rej], axis=0) / np.sqrt(c)
    if std_dev:
        return new_nu, new_data, new_std_dev
    return new_nu, new_data


def find_blind_channel(metadata):
    """
    This function identifies the blind detector and infers the phase-switch status.

    Parameters
    ----------

    data    : numpy array of shape (time*sampling_rate, 4),
              The output power of the 4 detectors.

    Returns
    -------
    -  the blind detector index
    -  the phase-switch status
    """

    phsw_state = metadata['phsw_state']
    if phsw_state in ('0101', '1010'):
        idx_blind = 3
    elif phsw_state in ('0110', '1001'):
        idx_blind = 0
    else:
        raise ValueError('invalid phase switch state: {0}'.format(phsw_state))

    return idx_blind, phsw_state


def get_central_nu_bandwidth(nu, data):
    """
    This function calculates the bandwidth and the central frequency of the input data.
    Definition are taken according to Bischoff and Newburgh's PhD Theses.

    Parameters
    ----------
    nu    : numpy array of shape (time*sampling_rate, ),
            The frequency data.
    data  : numpy array of shape (time*sampling_rate, 4),
            The output power of the 4 detectors
    Returns
    -------
    out : numpy array of shape (4, ), numpy array of shape (4, )

    """

    if not np.allclose((nu[1:] - nu[:-1]), (nu[1:] - nu[:-1])[0], rtol=1e-3):
        raise ValueError('The frequency steps are not uniform! Check out!')
    if data.shape[-1] == len(data):
        data = data[..., None]
    step = (nu[1:] - nu[:-1])[0]
    central_nu = np.sum(data * nu[..., None],
                        axis=0) / np.sum(data, axis=0)
    bandwidth = np.sum(data, axis=0)**2 * step / np.sum(data**2, axis=0)
    if bandwidth.shape[-1] == 1:
        return central_nu[0], bandwidth[0]
    return central_nu, bandwidth


def preliminary_plots(polarimeter_name, freq, data, output_path, pss, file_number, **kwargs):

    def axis_labels():
        plt.ylabel('Detector output' + r'$[ADU]$', fontsize=20)
        plt.xlabel('Frequency [GHz]', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    def save_plot(title, output_path):
        plot_file_path = os.path.join(output_path, plot_name + '.svg')
        plt.savefig(plot_file_path, bbox_inches='tight')
        log.info('Saving plot into file "%s"', plot_file_path)

    def plots(title, freq, data, legend_labels, output_path):
        plt.figure(figsize=(16, 9))
        plt.title(title, fontsize=22)
        plot = plt.plot(freq, data, **kwargs)
        plt.legend(plot, legend_labels, loc='best', fontsize=16)
        plt.grid(axis='y', linewidth=0.5)
        axis_labels()
        save_plot(title, output_path)

    plot_name = polarimeter_name + '_RFtest_' + pss + '_' + str(file_number)
    title = polarimeter_name + ' RFtest - ' + pss + '_' + str(file_number)
    plots(title, freq, data, NAMING_CONVENTION, output_path)


def final_plots(polarimeter_name, freq, norm_data, final_band, final_band_err,  output_path, **kwargs):

    def axis_labels():
        plt.ylabel('Detector output (normalized)', fontsize=20)
        plt.xlabel('Frequency [GHz]', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    def save_plot(title, output_path):
        plot_file_path = os.path.join(output_path, plot_name + '.svg')
        plt.savefig(plot_file_path, bbox_inches='tight')
        log.info('Saving plot into file "%s"', plot_file_path)

    def plots_all(title, freq, norm_data, final_band, legend_labels, output_path):
        plt.figure(figsize=(16, 9))
        plt.title(title, fontsize=22)
        plot = plt.plot(freq, norm_data, ':')
        plt.plot(freq, final_band, color='black')
        plt.legend(plot, legend_labels, loc='best', fontsize=16)
        plt.grid(axis='y', linewidth=0.5)
        axis_labels()
        save_plot(title, output_path)

    def plots_final(title, freq, norm_data, final_band, final_band_err, legend_labels, output_path):
        plt.figure(figsize=(16, 9))
        plt.title(title, fontsize=22)
        plot = plt.plot(freq, final_band, color='black')
        plt.fill_between(freq, final_band - final_band_err, final_band + final_band_err, alpha=0.2,
                         edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True)
        plt.legend(plot, legend_labels, loc='best', fontsize=16)
        plt.grid(axis='y', linewidth=0.5)
        axis_labels()
        save_plot(title, output_path)

    plot_name = polarimeter_name + '_RFtest_AllDetNorm'
    title = polarimeter_name + ' RFtest - All detector outputs (normalized)'
    labels = ['PW0/Q1 0101', 'PW1/U1 0101', 'PW2/U2 0101',
              'PW1/U1 0110', 'PW2/U2 0110', 'PW3/Q2 0110']
    plots_all(title, freq, norm_data, final_band,
              labels, output_path)

    plot_name = polarimeter_name + '_RFtest_FinalBand'
    title = polarimeter_name + ' RFtest - Final Band '
    labels = ['Final band']
    plots_final(title, freq, norm_data, final_band,
                final_band_err, labels, output_path)


def AnalyzeBandTest(polarimeter_name, file_name, output_path):
    """
       This function uses previous functions to analize a single bandpass test.

       Parameters
       ----------
       polarimeter_name
       file_name = name of the file to analize
       output_path = the path to the directory that will contain the report

       Returns
       -------
       out :
       test duration [s]
       frequency range
       phase-switch status
       frequencies: numpy array of shape (number of frequency steps, )
       selected data: numpy array of shape (number of frequency steps, 4)
       data normalized to range 0-1: numpy array of shape (number of frequency steps, 4)
       central frequency for each detector: numpy array of shape (4,)
       bandwidth for each detector: numpy array of shape (4,)
    """

    metadata, datafile = load_timestream(file_name)
    nu = datafile[-1]
    data = datafile[-3]

    log.info('File loaded, {0} samples found'.format(len(data[:, 0])))

    duration = len(data) / SAMPLING_FREQUENCY_HZ  # sec
    low_nu = np.min(nu[nu > 0])
    high_nu = np.max(nu)

    # Selecting data and removing the electronic offset

    nooffdata = remove_offset(nu, data, metadata)
    new_nu, new_data, new_std_dev = get_frequency_range_and_data(
        nu, nooffdata)

    # Setting to zero non physical values with negative gain
    new_data[new_data > 0] = 0

    # Computing central frequency and equivalent bandwidth for the four detectors
    central_nu_det, bandwidth_det = get_central_nu_bandwidth(
        new_nu, new_data)

    # Computing the average to get central frequency and equivalent bandwidth of the polarimeter (excluding the "blind" detector).
    # phase switch state '0101': PW3 is blind
    # phase switch state '0110': PW0 is blind

    # Finding the "blind" detector
    idx_blind, pss = find_blind_channel(metadata)
    mask = np.ones(4, dtype=bool)
    mask[idx_blind] = False

    central_nu_det = np.ma.masked_array(central_nu_det, mask=~mask)
    bandwidth_det = np.ma.masked_array(bandwidth_det, mask=~mask)

    # Normalizing data to range 0-1
    norm_data = new_data[:, mask] / \
        np.absolute(new_data[:, mask].min(axis=0))

    return duration, low_nu, high_nu, pss, new_nu, new_data, norm_data, central_nu_det, bandwidth_det


def build_dict_from_results(pol_name, duration, low_nu, high_nu, PSStatus, central_nu_det, bandwidth_det,
                            new_nu, final_band,
                            final_central_nu, final_central_nu_err,
                            final_bandwidth, final_bandwidth_err):
    results = {
        'polarimeter_name': pol_name,
        'title': 'Bandwidth test for polarimeter {0}'.format(pol_name),
        'low_frequency': low_nu,
        'high_frequency': high_nu,
        'sampling_frequency': SAMPLING_FREQUENCY_HZ,
        'test_duration': duration / 60 / 60,
        'central_nu_ghz': final_central_nu,
        'central_nu_err': final_central_nu_err,
        'bandwidth_ghz': final_bandwidth,
        'bandwidth_err': final_bandwidth_err,
    }

    detailed_results = []

    for j, pss in enumerate(PSStatus):
        cur_results = {}
        results['PSStatus' + '_' + str(j)] = pss
        cur_results['PSStatus'] = pss
        for i, nam in enumerate(NAMING_CONVENTION):
            nam = nam.replace("/", "")
            if(pss == '0101' and i == 3 or pss == '0110' and i == 0):
                central_nu_det[j, i] = 0
                bandwidth_det[j, i] = 0
            cur_results[nam] = {'central_nu': central_nu_det[j, i],
                                'bandwidth': bandwidth_det[j, i]}
        detailed_results.append(cur_results)

    results['detailed_results'] = detailed_results
    results['bandshape'] = {
        'frequency_ghz': list(new_nu),
        'response': list(final_band),
    }
    return results


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``list of files to analize``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter''')
    parser.add_argument('-FILE', action='append', dest='file_list', default=[],
                        help='Add all the files you want to analyze. USAGE: -FILE "file1.txt" -FILE "file2.txt" -FILE "file3.txt"')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the directory does not exist, it will be created''')
    return parser.parse_args()


def main():
    log.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=log.DEBUG)
    args = parse_arguments()

    log.info('Tuning radiometer "%s"', args.polarimeter_name)

    log.info('Writing the report into "%s"', args.output_path)

    # Creating the directory that will contain the report
    os.makedirs(args.output_path, exist_ok=True)

    norm_data_list, central_nu_det, bandwidth_det, PSStatus = list(), list(), list(), list()
    low_nu, high_nu = np.zeros(1), np.zeros(1)

    for i, file_name in enumerate(args.file_list):

        # Loading file
        log.info('Loading file "{0}"'.format(file_name))

        # Analyzing bandpass test for this file
        duration, low_nu, high_nu, pss, new_nu, new_data, norm_data, cf_det, bw_det = AnalyzeBandTest(
            args.polarimeter_name, file_name, args.output_path)

        # Producing preliminary plots
        preliminary_plots(args.polarimeter_name, new_nu,
                          new_data, args.output_path, pss, i)

        # Saving normalized data for both phase-switch status
        central_nu_det.append(cf_det)
        bandwidth_det.append(bw_det)
        norm_data_list.append(norm_data)
        PSStatus.append(pss)

        # Saving bandpass data of all detectors to .txt file
        np.savetxt(args.output_path + 'bandpass_data_' + pss + '_' + str(i) + '.txt',
                   np.column_stack([new_nu, new_data]), header='\t\t'.join(['nu', 'PW0/Q1', 'PW1/U1', 'PW2/U2', 'PW3/Q2']))

    log.info(
        'Computed bandwidth and central frequency for each detector for both phase-switch status')

    central_nu_det = np.array(central_nu_det)
    bandwidth_det = np.array(bandwidth_det)
    norm_data_All = np.column_stack(norm_data_list)

    All_central_nu, All_bandwidth = get_central_nu_bandwidth(
        new_nu, norm_data_All)

    # Computing the final band
    final_band = np.median(norm_data_All, axis=1)
    final_band_err = (np.percentile(
        norm_data_All, 97.7, axis=1) - np.percentile(norm_data_All, 2.7, axis=1)) / 2

    # Producing final plots
    final_plots(args.polarimeter_name, new_nu, norm_data_All,
                final_band, final_band_err, args.output_path)

    # Computing final central frequency and final bandwidth
    final_central_nu, final_bandwidth = get_central_nu_bandwidth(
        new_nu, final_band[:, None])

    # Computing errors for central frequency and bandwidth
    final_central_nu_err = (np.percentile(
        All_central_nu, 97.7) - np.percentile(All_central_nu, 2.7)) / 2
    final_bandwidth_err = (np.percentile(
        All_bandwidth, 97.7) - np.percentile(All_bandwidth, 2.7)) / 2

    log.info(
        'Computed final bandwidth and final central frequency')

    # Creating the report
    params = build_dict_from_results(
        args.polarimeter_name, duration, low_nu, high_nu, PSStatus, central_nu_det, bandwidth_det,
        new_nu, final_band,
        final_central_nu, final_central_nu_err,
        final_bandwidth, final_bandwidth_err)

    save_parameters_to_json(params=dict(params, **get_code_version_params()),
                            output_file_name=os.path.join(args.output_path,
                                                          'bandwidth_results.json'))

    create_report(params=params,
                  md_template_file='bandwidth.md',
                  md_report_file='bandwidth_report.md',
                  html_report_file='bandwidth_report.html',
                  output_path=args.output_path)


if __name__ == '__main__':
    main()
