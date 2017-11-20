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

DEFAULT_LEFT_FREQ = 0.05 # Hz
DEFAULT_RIGHT_FREQ = 1 # Hz

DEM = ['DEM0/Q1', 'DEM1/U1', 'DEM2/U2', 'DEM3/Q2']
PWR = ['PWR0/Q1', 'PWR1/U1', 'PWR2/U2', 'PWR3/Q2']
STOKES = ['I', 'Q', 'U']

FIGSIZE = (10, 7)

def get_stokes(pwr_data, dem_data):
    """
    This function returns the Stokes parameters I, Q, U starting from the demodulated signal of the 
    4 detectors. 

    Parameters
    ----------
    pwr_data : numpy array of shape (time*sampling_rate, 4),
               The total power signal of the 4 detectors.
    dem_data : numpy array of shape (time*sampling_rate, 4),
               The demodulated signal of the 4 detectors.

    Returns
    -------
    out : numpy array of shape (time*sampling_rate, ), numpy array of shape (time*sampling_rate, ),
          numpy array of shape (time*sampling_rate, )
    """
    I = np.sum(pwr_data, axis=1) / 4
    Q = (dem_data[:, 0] - dem_data[:, 3]) / 2
    U = (dem_data[:, 1] - dem_data[:, 2]) / 2
    return np.column_stack((I, Q, U))


def get_fft(sampling_frequency, data, n_chunks, detrend='linear', **kwargs):
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
    detrend            : str or function or `False`, optional
                         Specifies how to detrend each segment. If `detrend` is a string, it is 
                         passed as the `type` argument to the `detrend` function. If it is a 
                         function, it takes a segment and returns a detrended segment. If `detrend` 
                         is `False`, no detrending is done. Defaults to 'linear'.
 
    Returns
    -------
    out : numpy array of shape (time*sampling_rate // 2, ), 
          numpy array of shape (time*sampling_rate // 2, 4)
    """
    # cut the last data if the number of data is odd
    cdata = data[0:(len(data)//2)*2]
    
    N = len(cdata)
    freq, fft = signal.welch(cdata, sampling_frequency, nperseg=N/n_chunks, axis=0, detrend=detrend,
                             **kwargs)
    return freq[1:], fft[1:]


def get_noise_characteristics(freq, fft, left_freq, right_freq, totalPWR=False):
    """
    This functions returns an estimation of the slope of the pink spectrum, of the white noise level
    and of the knee frequency of noise power spectrum. 
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
    totalPWR   : string or boolean,
                 If True only the slope of the pink spectrum is returned. If `first`... 
                 Defaults to False.

    Returns
    -------
    out : numpy array of shape (4, ), numpy array of shape (3, 4), numpy array of shape (4, ),
          numpy array of shape (4, ) 
          or, if totalPWR is True: numpy array of shape (4, )
    """
    def get_parameters(freq, fft, left_freq, right_freq, totalPWR):
        if totalPWR:
            left_freq += 5
        f_idx_left = np.argwhere((freq - left_freq) > 0)[0, 0] 
        f_idx_right = np.argwhere((freq - right_freq) > 0)[0, 0]

        # fit the left part of the spectrum with a line
        ab, cov = np.polyfit(np.log(freq[:f_idx_left]), np.log(fft[:f_idx_left]), 1, cov=True)
        a, b = ab
        delta_a, delta_b = (cov[0, 0], cov[1, 1])
        
        # calculate median value of the right part of the spectrum
        if totalPWR:
            c, fknee_ = (np.full_like(a, np.NaN), np.full_like(a, np.NaN))
        else:
            c = np.median(np.log(fft[f_idx_right:]), axis=0)
            fknee_ = np.exp((c - b) / a)
        fit_par = np.row_stack([a, b, c])

        # uncertanties estimation
        def get_right_number_of_decimals(x, delta_x):
            def get_new_x(x, new_num_dec):
                new_x = np.array([np.around(x[i], decimals=dec) for i, dec in
                                  enumerate(new_num_dec)])
                new_x[new_num_dec==1] = np.around(new_x[new_num_dec==1], decimals=1)
                return new_x
            new_num_dec = np.int_(np.ceil(np.abs(np.log10(delta_x))))
            return get_new_x(x, new_num_dec), get_new_x(delta_x, new_num_dec)

        delta_c = np.sum(np.abs(np.log(fft[f_idx_right:]) - c), axis=0) / len(fft[f_idx_right:])
        delta_fknee_ = fknee_ / np.abs(a) * np.sqrt(((c - b) / a)**2 * delta_a**2 + delta_b**2 +
                                                    delta_c**2)
        delta_median_ = np.exp(c) * delta_c

        if totalPWR:
            fknee, delta_fknee = (np.full_like(fknee_, np.NaN), np.full_like(delta_fknee_, np.NaN))
            median, delta_median = (np.full_like(c, np.NaN), np.full_like(delta_median_, np.NaN))
        else:
            fknee, delta_fknee = get_right_number_of_decimals(fknee_, delta_fknee_)
            median, delta_median = get_right_number_of_decimals(np.exp(c), delta_median_)
        slope, delta_slope = get_right_number_of_decimals(-a, delta_a)
        
        return fit_par, fknee, delta_fknee, slope, delta_slope, median, delta_median
                
    
    if totalPWR == 'stokes':
        (fit_parI, fkneeI, delta_fkneeI, alphaI, delta_alphaI, WN_levelI,
         delta_WN_levelI) = get_parameters(
            freq, fft[:, 0][..., None], left_freq, right_freq, totalPWR=True)
        (fit_parQU, fkneeQU, delta_fkneeQU, alphaQU, delta_alphaQU, WN_levelQU,
         delta_WN_levelQU) = get_parameters(
             freq, fft[:, 1:], left_freq, right_freq, totalPWR=False)
        return (np.column_stack((fit_parI, fit_parQU)), np.append(fkneeI, fkneeQU),
                np.append(delta_fkneeI, delta_fkneeQU), np.append(alphaI, alphaQU),
                np.append(delta_alphaI, delta_alphaQU), np.append(WN_levelI, WN_levelQU),
                np.append(delta_WN_levelI, delta_WN_levelQU))
    return get_parameters(freq, fft, left_freq, right_freq, totalPWR)


def create_plots(polarimeter_name, freq, fftDEM, fit_parDEM, labelsDEM, fftPWR, fit_parPWR,
                 labelsPWR, fftIQU, fit_parIQU, labelsSTOKES, output_path, **kwargs):
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
                       It is the power spectrum of the demodulated data. 
    fit_parDEM       : numpy array of shape (3, 4),
                       The parameters for the fit of the DEM outputs.
    labelsDEM        : list of string of len(3),
                       The labels of the demodulated data that will be shown.
    fftPWR           : numpy array of shape (time*sampling_rate // 2, 4),
                       It is the power spectrum of the total power data. 
    fit_parPWR       : numpy array of shape (3, 4),
                       The parameters for the fit of the PWR outputs.
    labelsPWR        : list of string of len(3),
                       The labels of the total power data that will be shown.
    fftIQU           : numpy array of shape (time*sampling_rate // 2, 3),
                       It is the power spectrum of the combined data to form I, Q, U. 
    fit_parIQU       : numpy array of shape (3, 3),
                       The parameters for the fit of the combined data to form I, Q, U.
    labelsIQU        : list of string of len(3),
                       The labels of the combined data to form I, Q, U that will be shown.
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
        plt.figure(figsize=FIGSIZE)
        plt.title(title, fontsize=22)
        data = plt.loglog(freq, fft, **kwargs)
        plt.legend(data, legend_labels, loc='best', fontsize=16)
        axis_labels()
        save_plot(replace(title), output_path)
        
    def single_plots(title_, freq, fft, fit_par, output_path):
        for i in range(fft.shape[-1]):
            plt.figure(figsize=FIGSIZE)
            title = polarimeter_name + ' PSD - ' + title_[i]
            plt.title(title, fontsize=22)
            plt.loglog(freq, fft[:, i], **kwargs)
            a, b, c = fit_par[:, i]
            plt.loglog(freq, freq**a * np.e**b, 'r', lw=2)
            plt.loglog(freq, np.full_like(freq, np.exp(c)), 'r', lw=2)
            axis_labels()
            save_plot(replace(title), output_path)

    # Plot DEM outputs separately        
    single_plots(labelsDEM, freq, fftDEM, fit_parDEM, output_path)

    # Plot I, Q, U separately 
    single_plots(labelsSTOKES, freq, fftIQU, fit_parIQU, output_path)

    # Plot DEM and PWR outputs together        
    title = polarimeter_name + ' PSD - all detector outputs'
    legend_labels = labelsDEM + labelsPWR
    comulative_plots(title, freq, np.column_stack((fftDEM, fftPWR)), legend_labels, output_path)

    # Plot I, Q, U together
    title = polarimeter_name + ' PSD - I Q U'
    comulative_plots(title, freq, fftIQU, labelsSTOKES, output_path)

    # Plot PWR and I
    title = polarimeter_name + ' PSD - ' + ' '.join(labelsPWR) + ' ' + labelsSTOKES[0]
    fftQ1Q2Q = np.column_stack((fftPWR, fftIQU[:, 0]))
    legend_labels = labelsPWR + [labelsSTOKES[0]]
    comulative_plots(title, freq, fftQ1Q2Q, legend_labels, output_path)
    
    # Plot Q1, Q2 and Q
    title = polarimeter_name + ' PSD - ' + ' '.join(labelsDEM[0::3]) + ' ' + labelsSTOKES[1]
    fftQ1Q2Q = np.column_stack((fftDEM[:, 0], fftDEM[:, 3], fftIQU[:, 1]))
    legend_labels = labelsDEM[0::3] + [labelsSTOKES[1]]
    comulative_plots(title, freq, fftQ1Q2Q, legend_labels, output_path)

    # Plot U1, U2 and U
    title = polarimeter_name + ' PSD - ' + ' '.join(labelsDEM[1:3]) + ' ' + labelsSTOKES[2]
    fftU1U2U = np.column_stack((fftDEM[:, 1], fftDEM[:, 2], fftIQU[:, 2]))
    legend_labels = labelsDEM[1:3] + [labelsSTOKES[2]]
    comulative_plots(title, freq, fftU1U2U, legend_labels, output_path)


def get_y_intercept_1_f_reduction(freq, fit_par):
    '''Compute the value of the y-intercept and return an estimation of the mean value of 1/f 
       reduction.

    Parameters
    ----------
    freq       : numpy array of shape (time*sampling_rate // 2, ),
                 The frequency domain ranging from 1 / (total duration of the test) to (sampling 
                 rate / 2) Hz.
    fit_parDEM : numpy array of shape (3, 4),
                 The parameters for the fit of the DEM outputs.

    '''
    y_intercepts = np.exp(fit_par[0] * np.log(freq.min()) + fit_par[1])
    reduction1_f = np.mean([y_intercepts[0] / y_intercepts[1], y_intercepts[0] / y_intercepts[2]])
    return np.int(np.log10(reduction1_f))


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``input_file_path``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--detrend', dest='detrend',
                        type=str, default='linear',
                        help='''Specifies how to detrend each segment
                        (default: linear)''')
    parser.add_argument('--1/f-upper-frequency', dest='left_freq',
                        type=float, default=DEFAULT_LEFT_FREQ,
                        help='''Upper frequency for 1/f estimation
                        (default: {0})'''.format(DEFAULT_LEFT_FREQ))
    parser.add_argument('--WN-lower-frequency', dest='right_freq',
                        type=float, default=DEFAULT_RIGHT_FREQ,
                        help='''Lower frequency for white noise estimation
                        (default: {0})'''.format(DEFAULT_RIGHT_FREQ))
    parser.add_argument('--number-of-chunks', dest='n_chunks',
                        type=int,
                        help='''Number of chunks used for the estimation of the PSD 
                        (default: duration [hours] * 12)''')
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter (any text string
                        is ok, it is used in the reports)''')
    parser.add_argument('input_file_path', type=str,
                        help='''Name of the file containing the data being saved''')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the path does not exist, it will be created''')
    return parser.parse_args()


def build_dict_from_results(pol_name, duration, left_freq, right_freq, n_chuncks, detrend,
                            reduction1_f, fkneeDEM, delta_fkneeDEM, alphaDEM, delta_alphaDEM,
                            WN_levelDEM, delta_WN_levelDEM, fkneePWR, delta_fkneePWR, alphaPWR,
                            delta_alphaPWR, WN_levelPWR, delta_WN_levelPWR, fkneeIQU,
                            delta_fkneeIQU, alphaIQU, delta_alphaIQU, WN_levelIQU,
                            delta_WN_levelIQU):
    results = {
        'polarimeter_name': pol_name,
        'title': 'Noise characteristics of polarimeter {0}'.format(pol_name),
        'sampling_frequency_hz': SAMPLING_FREQUENCY_HZ,
        'test_duration_hz': duration / 60 / 60,
        'left_freq_hz': left_freq,
        'right_freq_hz': right_freq,
        'n_chunks': n_chuncks,
        'detrend': detrend,
        'reduction_factor_1f': reduction1_f}
    
    for i, nam in enumerate(DEM):
        nam = nam.replace("/", "")
        results[nam] = {'f_knee_hz' : fkneeDEM[i],
                        'delta_f_knee_hz' : delta_fkneeDEM[i],
                        'slope' : alphaDEM[i],
                        'delta_slope' : delta_alphaDEM[i],
                        'WN_level_adu2_hz' : WN_levelDEM[i],
                        'delta_WN_level_adu2_hz' : delta_WN_levelDEM[i]}

    for i, pwr in enumerate(PWR):
        pwr = pwr.replace("/", "")
        results[pwr] = {'f_knee_hz' : fkneePWR[i],
                        'delta_f_knee_hz' : delta_fkneePWR[i],
                        'slope' : alphaPWR[i],
                        'delta_slope' : delta_alphaPWR[i],
                        'WN_level_adu2_hz' : WN_levelPWR[i],
                        'delta_WN_level_adu2_hz' : delta_WN_levelPWR[i]}

    for i, stokes in enumerate(STOKES):
        results[stokes] = {'f_knee_hz' : fkneeIQU[i],
                        'delta_f_knee_hz' : delta_fkneeIQU[i],
                        'slope' : alphaIQU[i],
                        'delta_slope' : delta_alphaIQU[i],
                        'WN_level_adu2_hz' : WN_levelIQU[i],
                        'delta_WN_level_adu2_hz' : delta_WN_levelIQU[i]}
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
    dataDEM = np.loadtxt(args.input_file_path, skiprows=1, usecols=(3, 4, 5, 6))
    durationDEM = len(dataDEM) / SAMPLING_FREQUENCY_HZ # sec
    dataPWR = np.loadtxt(args.input_file_path, skiprows=1, usecols=(7, 8, 9, 10))
    durationPWR = len(dataPWR) / SAMPLING_FREQUENCY_HZ # sec

    assert durationPWR == durationDEM
    duration = durationDEM

    if args.n_chunks is None:
        args.n_chunks = np.int(duration / 60 / 60 * 12) # each chunk lasts 5 minutes by default

    log.info('File loaded, {0} samples found'.format(duration * SAMPLING_FREQUENCY_HZ))

    # Calculate the PSD
    log.info(
        'Computing PSD with number-of-chunks {0}, 1/f-upper-frequency {1}, WN-lower-frequency{2}' +
        ', detrend {3}'.format(args.n_chunks, args.left_freq, args.right_freq, args.detrend))
    
    freq, fftDEM = get_fft(SAMPLING_FREQUENCY_HZ, dataDEM, args.n_chunks, detrend=args.detrend)   
    (fit_parDEM, fkneeDEM, delta_fkneeDEM, alphaDEM, delta_alphaDEM, WN_levelDEM,
     delta_WN_levelDEM) = get_noise_characteristics(
        freq, fftDEM, args.left_freq, args.right_freq)
    [log.info('Computed fknee, alpha, WN_level for' + nam + ' outputs') for nam in DEM]

    fftPWR = get_fft(SAMPLING_FREQUENCY_HZ, dataPWR, args.n_chunks, detrend=args.detrend)[-1]   
    (fit_parPWR, fkneePWR, delta_fkneePWR, alphaPWR, delta_alphaPWR, WN_levelPWR,
     delta_WN_levelPWR) = get_noise_characteristics(
        freq, fftPWR, args.left_freq, args.right_freq, totalPWR=True)
    [log.info('Computed alpha for' + pwr + ' outputs') for pwr in PWR]
    
    # Calculate the PSD for the combinations of the 4 detector outputs that returns I, Q, U
    IQU = get_stokes(dataPWR, dataDEM)
    fftIQU = get_fft(SAMPLING_FREQUENCY_HZ, IQU, args.n_chunks, detrend=args.detrend)[-1]
    (fit_parIQU, fkneeIQU, delta_fkneeIQU, alphaIQU, delta_alphaIQU, WN_levelIQU,
     delta_WN_levelIQU) = get_noise_characteristics(
        freq, fftIQU, args.left_freq, args.right_freq, totalPWR='stokes')
    log.info('Computed fknee, alpha, WN_level for I, Q, U')

    # Get an approximate estimation of the 1/f reduction factor
    reduction1_f = get_y_intercept_1_f_reduction(freq, fit_parIQU)  
    
    # Produce the plots
    create_plots(args.polarimeter_name, freq, fftDEM, fit_parDEM, DEM, fftPWR, fit_parPWR, PWR,
                 fftIQU, fit_parIQU, STOKES, args.output_path)
        
    params = build_dict_from_results(args.polarimeter_name, duration, args.left_freq,
                                     args.right_freq, args.n_chunks, args.detrend, reduction1_f,
                                     fkneeDEM, delta_fkneeDEM, alphaDEM, delta_alphaDEM,
                                     WN_levelDEM, delta_WN_levelDEM, fkneePWR, delta_fkneePWR,
                                     alphaPWR, delta_alphaPWR, WN_levelPWR, delta_WN_levelPWR,
                                     fkneeIQU, delta_fkneeIQU, alphaIQU, delta_alphaIQU,
                                     WN_levelIQU, delta_WN_levelIQU)

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
