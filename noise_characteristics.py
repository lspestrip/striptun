#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''Estimates the noise characteristics of a given polarimeter.'''

from json_save import save_parameters_to_json
from argparse import ArgumentParser
from scipy import signal
from reports import create_report
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import os

SAMPLING_FREQUENCY_HZ = 25.0 # Hz

DEFAULT_LEFT_FREQ = 0.033 # Hz
DEFAULT_RIGHT_FREQ = 1 # Hz

N_CHUNKS = 6

NAMING_CONVENTION = ['DEM0/Q1', 'DEM1/U1', 'DEM2/U2', 'DEM3/Q2']
STOKES = ['I', 'Q', 'U']


def get_stokes(data):
    """
    This function returns the Stokes parameters I, Q, U starting from the demodulated signal of the 
    4 detectors. 

    Parameters
    ----------
    data : numpy array of shape (time*sampling_rate, 4),
           The demodulated signal of the 4 detectors.

    Returns
    -------
    out : numpy array of shape (time*sampling_rate, ), numpy array of shape (time*sampling_rate, ),
          numpy array of shape (time*sampling_rate, )
    """
    I = np.sum(data, axis=1)
    Q = (data[:, 0] - data[:, 3]) / 2
    U = (data[:, 1] - data[:, 2]) / 2
    return np.column_stack((I, Q, U))


def get_fft(sampling_frequency, data, n_chunks, **kwargs):
    """
    This function returns the power spectral density (PSD) of a given array of data. The function 
    exploits the scipy function: signal.welch.

    Parameters
    ----------
    sampling_frequency : numpy array of shape (time*sampling_rate, ),
                         The temporal data sample.
    data               : numpy array of shape (time*sampling_rate, 4),
                         It is the demodulated output power of the 4 detectors.
    n_chunks           : integer, 
                         The number of chuncks in which the data will be devided. 

    Returns
    -------
    out : numpy array of shape (time*sampling_rate // 2, ), 
          numpy array of shape (time*sampling_rate // 2, 4)
    """
    # cut the last data if the number of data is odd
    cdata = data[0:(len(data)//2)*2]
    
    N = len(cdata)
    freq, fft = signal.welch(cdata, sampling_frequency, nperseg=N/n_chunks, axis=0, **kwargs)
    return freq[1:], fft[1:]


def get_fknee(freq, fft, left_freq, right_freq):
    """
    This functions returns an estimation of the knee frequency of the noise power spectrum. 
    The left and right part of the spectrum will be fitted with two lines (in logarithmic
    coordinates) whose parameters will be returned as well. In particular, it will return the slope 
    of the left fit (alpha) and the white noise (WN) median level. 

    Parameters
    ----------
    freq       : numpy array of shape (time*sampling_rate // 2, ),
                 The frequency domain ranging from 1 / (total duration of the test) to (sampling 
                 rate / 2) Hz.
    fft        : numpy array of shape (time*sampling_rate // 2, ) or 
                 (time*sampling_rate // 2, ),
                 It is the power spectrum of the data.
    left_freq  : float,
                 The frequency below which is fitted the left part of the plot.  
    right_freq : float,
                 The frequency above which is fitted the right part of the plot.

    Returns
    -------
    out : numpy array of shape (4, ), numpy array of shape (3, 4), numpy array of shape (4, ),
          numpy array of shape (4, )
    """
   
    f_idx_left = np.argwhere((freq - left_freq) > 0)[0, 0] 
    f_idx_right = np.argwhere((freq - right_freq) > 0)[0, 0]

    # fit the left part of the spectrum with a line
    a, b = np.polyfit(np.log(freq[:f_idx_left]), np.log(fft[:f_idx_left]), 1)
    
    # calculate median value of the right part of the spectrum
    c = np.median(np.log(fft[f_idx_right:]), axis=0)
    
    fknee = np.exp((c - b) / a)
    fit_par = np.row_stack([a, b, c])
    return fit_par, fknee, -a, np.e**c 


def create_plots(polarimeter_name, freq, fftDEM, fftIQU, fit_parDEM, fit_parIQU, output_path,
                 **kwargs):
    """
    This function shows the fft data.

    Parameters
    ----------
    polarimeter_name : string,
                       The polarimeter name.
    freq             : numpy array of shape (time*sampling_rate // 2, ),
                       The frequency domain ranging from 1 / (total duration of the test) to 
                       (sampling rate / 2) Hz. It will be plotted on the x axis.  
    fftDEM           : numpy array of shape (time*sampling_rate // 2, 4),
                       It is the power spectrum of the data. 
    fftIQU           : numpy array of shape (time*sampling_rate // 2, 3),
                       It is the power spectrum of the combined data to form I, Q, U. 
    fit_parDEM       : numpy array of shape (3, 4),
                       The parameters for the fit of the DEM outputs.
    fit_parIQU       : numpy array of shape (3, 3),
                       The parameters for the fit of the combined data to form I, Q, U.
    output_path      : string,
                       Path to the directory that will contain the report.
    """ 

    def axis_labels():
        plt.ylabel('Power Spectrum ' + r'$[ADU^2 / Hz]$', fontsize=20)
        plt.xlabel('Frequency [Hz]', fontsize=20)
        plt.xlim(freq[0], freq[-1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    def save_plot(title, output_path):
        plot_file_path = os.path.join(output_path, title + '.svg')
        plt.savefig(plot_file_path, bbox_inches='tight')
        log.info('Saving plot into file "%s"', plot_file_path)

    def replace(title):
        title = title.replace("- ", "")
        title = title.replace(" ", "_")
        title = title.replace("/", "_")
        return title
    
    def comulative_plots(title, freq, fft, legend_labels, output_path):
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=22)
        data = plt.loglog(freq, fft, **kwargs)
        plt.legend(data, legend_labels, loc='best', fontsize=16)
        axis_labels()
        save_plot(replace(title), output_path)
        
    def single_plots(title_, freq, fft, fit_par, output_path):
        for i in range(fft.shape[-1]):
            plt.figure(figsize=(8, 6))
            title = polarimeter_name + ' PSD - ' + title_[i]
            plt.title(title, fontsize=22)
            plt.loglog(freq, fft[:, i], **kwargs)
            a, b, c = fit_par[:, i]
            plt.loglog(freq, freq**a * np.e**b, 'r', lw=2)
            plt.loglog(freq, np.full_like(freq, np.exp(c)), 'r', lw=2)
            axis_labels()
            save_plot(replace(title), output_path)

    # Plot DEM outputs separately        
    single_plots(NAMING_CONVENTION, freq, fftDEM, fit_parDEM, output_path)

    # Plot DEM outputs together        
    title = polarimeter_name + ' PSD - all detectors'
    comulative_plots(title, freq, fftDEM, NAMING_CONVENTION, output_path)

    # Plot I, Q, U separately 
    single_plots(STOKES, freq, fftIQU, fit_parIQU, output_path)

    # Plot I, Q, U together
    title = polarimeter_name + ' PSD - I Q U'
    comulative_plots(title, freq, fftIQU, STOKES, output_path)

    # Plot Q1, Q2 and Q
    title = polarimeter_name + ' PSD - DEM0/Q1 DEM3/Q2 Q'
    fftQ1Q2Q = np.column_stack((fftDEM[:, 0], fftDEM[:, 3], fftIQU[:, 1]))
    legend_labels = ['DEM0/Q1', 'DEM3/Q2', 'Q']
    comulative_plots(title, freq, fftQ1Q2Q, legend_labels, output_path)

    # Plot U1, U2 and U
    title = polarimeter_name + ' PSD - DEM1/U1 DEM2/U2 U'
    fftU1U2U = np.column_stack((fftDEM[:, 1], fftDEM[:, 2], fftIQU[:, 2]))
    legend_labels = ['DEM1/U1', 'DEM2/U2', 'U']
    comulative_plots(title, freq, fftU1U2U, legend_labels, output_path)


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``input_file_path``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--1/f-upper-frequency', dest='left_freq',
                        type=float, default=DEFAULT_LEFT_FREQ,
                        help='''Upper frequency for 1/f estimation
                        (default: {0})'''.format(DEFAULT_LEFT_FREQ))
    parser.add_argument('--WN-lower-frequency', dest='right_freq',
                        type=float, default=DEFAULT_RIGHT_FREQ,
                        help='''Lower frequency for white noise estimation
                        (default: {0})'''.format(DEFAULT_RIGHT_FREQ))
    parser.add_argument('--number-of-chunks', dest='n_chunks',
                        type=int, default=N_CHUNKS,
                        help='''Number of chunks used for the estimation of the PSD 
                        (default: {0})'''.format(N_CHUNKS))
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter (any text string
                        is ok, it is used in the reports)''')
    parser.add_argument('input_file_path', type=str,
                        help='''Name of the file containing the data being saved''')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the path does not exist, it will be created''')
    return parser.parse_args()


def build_dict_from_results(pol_name, duration, left_freq, right_freq, n_chuncks, fkneeDEM,
                            alphaDEM, WN_levelDEM, fkneeIQU, alphaIQU, WN_levelIQU):
    results = {
        'polarimeter_name': pol_name,
        'title': 'Noise characteristics of polarimeter {0}'.format(pol_name),
        'sampling_frequency': SAMPLING_FREQUENCY_HZ,
        'test_duration' : duration / 60 / 60,
        'left_freq' : left_freq,
        'right_freq': right_freq,
        'n_chunks': n_chuncks}
    
    for i, nam in enumerate(NAMING_CONVENTION):
        nam = nam.replace("/", "")
        results[nam] = {'f_knee' : fkneeDEM[i] * 1000,
                        'slope' : alphaDEM[i],
                        'WN_level' : WN_levelDEM[i]}

    for i, stokes in enumerate(STOKES):
        results[stokes] = {'f_knee' : fkneeIQU[i] * 1000,
                           'slope' : alphaIQU[i],
                           'WN_level' : WN_levelIQU[i]}

    return results

            
def main():

    log.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=log.DEBUG)
    args = parse_arguments()

    log.info('Tuning radiometer "%s"', args.polarimeter_name)
    log.info('Reading data from file "%s"', args.input_file_path)
    log.info('Writing the report into "%s"', args.output_path)

    # Create the directory that will contain the report
    os.makedirs(args.output_path, exist_ok=True)

    # Load from the text file only the columns containing the output of the four detectors
    log.info('Loading file "{0}"'.format(args.input_file_path))
    data = np.loadtxt(args.input_file_path, skiprows=1, usecols=(3, 4, 5, 6))
    duration = len(data) / SAMPLING_FREQUENCY_HZ # sec

    log.info('File loaded, {0} samples found'.format(len(data[:, 0])))

    # Calculate the PSD
    log.info(
        'Computing PSD with number-of-chunks {0}, 1/f-upper-frequency {1}, WN-lower-frequency{2}'
        .format(args.n_chunks, args.left_freq, args.right_freq))
    
    freq, fftDEM = get_fft(SAMPLING_FREQUENCY_HZ, data, args.n_chunks)   
    fit_parDEM, fkneeDEM, alphaDEM, WN_levelDEM = get_fknee(
        freq, fftDEM, args.left_freq, args.right_freq)
    [log.info('Computed fknee, alpha, WN_level for' + nam + ' outputs') for nam in
     NAMING_CONVENTION]
    
    # Calculate the PSD for the combinations of the 4 detector outputs that returns I, Q, U
    IQU = get_stokes(data)
    fftIQU = get_fft(SAMPLING_FREQUENCY_HZ, IQU, args.n_chunks)[-1]
    fit_parIQU, fkneeIQU, alphaIQU, WN_levelIQU = get_fknee(
        freq, fftIQU, args.left_freq, args.right_freq)
    log.info('Computed fknee, alpha, WN_level for I, Q, U')
    
    # Produce the plots
    create_plots(args.polarimeter_name, freq, fftDEM, fftIQU, fit_parDEM, fit_parIQU,
                 args.output_path)
        
    params = build_dict_from_results(args.polarimeter_name, duration, args.left_freq,
                                     args.right_freq, args.n_chunks, fkneeDEM, alphaDEM,
                                     WN_levelDEM, fkneeIQU, alphaIQU, WN_levelIQU)

    save_parameters_to_json(params=params,
                            output_file_name=os.path.join(args.output_path,
                                                          'noise_characteristics_results.json'))

    create_report(params=params,
                  md_template_file='noise_characteristics.md',
                  md_report_file='noise_characteristics_report.md',
                  html_report_file='noise_characteristics_report.html',
                  output_path=args.output_path)

    
if __name__ == '__main__':
    main()
