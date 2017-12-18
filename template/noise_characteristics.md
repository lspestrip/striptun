## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise characteristics for
the Strip polarimeter **${polarimeter_name}** and refers to the test ["${input_file_path}"](${input_file_path}).

The report has been generated on ${analysis_date} using striptun
v${striptun_version} (commit
[${latest_git_commit[0:6]}](https://github.com/lspestrip/striptun/commit/${latest_git_commit})).

This test has been performed with a **sampling frequency of ${sampling_frequency_hz} [Hz]**
 and **lasted ${'{0:.2f}'.format(test_duration_hz)} [hours]**.

<h2>Results</h2>

The Power Spectral Densities (PSD) of the four detector outputs and of 
thier opportune combinations I, Q, U have been estimated using Welch's method as described in 
"Numerical recipes - The art of Scientific Computing", W.H. Press et al., Third edition
, pp.652-662. The function `signal.welch` of the Python library `scipy` has been used to 
implement it in the code.

The raw data [ADU] have been divided by the detector gain obtained from ["${gains_file_path}"](${input_file_path}) to obtain calibrated data [K]. For the total power data the detector offsets have been subtracted before dividing.



The original data samples have been divided into **${n_chunks} segments** of equal length. 
For each of them, the periodogram has been estimated and then the average at each 
frequency has been computed. Each segment has been **detrended** by subtracting a *${detrend} fit* of the data.


<h2>White noise and 1/f estimation</h2>

The *white noise level* has been estimated calculating the median value of the spectrum starting 
**from ${right_freq_hz} [Hz]** up to the higher frequency. 
The *slope of the pink spectrum* has been extracted with a linear fit of the left part of the spectrum **until ${left_freq_hz} [Hz]**. 
The *knee frequency* has been estimated by doing the intersection between the linear fit of the left part of the spectrum 
and the median of the right part of the spectrum.

<h3>Demodolated detector outputs</h3>

In this section are reported the results of the analysis for the four demodulated detector 
outputs.

![](${polarimeter_name}_PSD_DEM0_Q1.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM1_U1.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM2_U2.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM3_Q2.svg){: class="plot"} 


DETECTOR  | f knee [mHz]         | alpha [#]           | white noise level [mK<sup>2</sup>/Hz]*
--------- |:--------------------:|:-------------------:|:----------------------:
DEM0/Q1   | ${'{:0.0f}'.format(DEM0Q1['f_knee_hz'] * 1000)} &#177; ${'{:0.0f}'.format(DEM0Q1['delta_f_knee_hz'] * 1000)} | ${DEM0Q1['slope']} &#177; ${DEM0Q1['delta_slope']} | ${DEM0Q1['WN_level_K2_hz'] * 1000000} &#177; ${DEM0Q1['delta_WN_level_K2_hz'] * 1000000}
DEM1/U1   | ${'{:0.0f}'.format(DEM1U1['f_knee_hz'] * 1000)} &#177; ${'{:0.0f}'.format(DEM1U1['delta_f_knee_hz'] * 1000)} | ${DEM1U1['slope']} &#177; ${DEM1U1['delta_slope']} | ${DEM1U1['WN_level_K2_hz'] * 1000000} &#177; ${DEM1U1['delta_WN_level_K2_hz'] * 1000000}
DEM2/U2   | ${'{:0.0f}'.format(DEM2U2['f_knee_hz'] * 1000)} &#177; ${'{:0.0f}'.format(DEM2U2['delta_f_knee_hz'] * 1000)} | ${DEM2U2['slope']} &#177; ${DEM2U2['delta_slope']} | ${DEM2U2['WN_level_K2_hz'] * 1000000} &#177; ${DEM2U2['delta_WN_level_K2_hz'] * 1000000}
DEM3/Q2   | ${'{:0.0f}'.format(DEM3Q2['f_knee_hz'] * 1000)} &#177; ${'{:0.0f}'.format(DEM3Q2['delta_f_knee_hz'] * 1000)} | ${DEM3Q2['slope']} &#177; ${DEM3Q2['delta_slope']} | ${DEM3Q2['WN_level_K2_hz'] * 1000000} &#177; ${DEM3Q2['delta_WN_level_K2_hz'] * 1000000}


*To estimate the uncertainty on the white noise level has been used the median deviation.

<h3>Stokes parameters signals</h3>

In this section are reported the results of the analysis for the opportune combinations of the four 
detector outputs, which provides the Stokes parameters.

The following combinations of the detector outputs have been used:

$$ \begin{align} I &= \frac{PWR0/Q1 + PWR1/U1 + PWR2/U2 + PWR3/Q2}4, \newline
Q &= \frac{DEM0/Q1 - DEM3/Q2}2,  \newline
U &= \frac{DEM1/U1 - DEM2/U2}2. \end{align} 
$$

![](${polarimeter_name}_PSD_I.svg){: class="plot"}
![](${polarimeter_name}_PSD_Q.svg){: class="plot"}
![](${polarimeter_name}_PSD_U.svg){: class="plot"}

SIGNAL  | f knee [mHz]   | alpha [#]     | white noise level [mK<sup>2</sup>/Hz]* 
------- |:--------------:|:-------------:|:----------------------:
I       | ${'{:0.0f}'.format(I['f_knee_hz']* 1000)} &#177; ${'{:0.0f}'.format(I['delta_f_knee_hz'] * 1000)} | ${I['slope']} &#177; ${I['delta_slope']} | ${I['WN_level_K2_hz'] * 1000000} &#177; ${I['delta_WN_level_K2_hz'] * 1000000}
Q       | ${'{:0.0f}'.format(Q['f_knee_hz']* 1000)} &#177; ${'{:0.0f}'.format(Q['delta_f_knee_hz'] * 1000)} | ${Q['slope']} &#177; ${Q['delta_slope']} | ${Q['WN_level_K2_hz'] * 1000000} &#177; ${Q['delta_WN_level_K2_hz'] * 1000000}
U       | ${'{:0.0f}'.format(U['f_knee_hz']* 1000)} &#177; ${'{:0.0f}'.format(U['delta_f_knee_hz'] * 1000)} | ${U['slope']} &#177; ${U['delta_slope']} | ${U['WN_level_K2_hz'] * 1000000} &#177; ${U['delta_WN_level_K2_hz'] * 1000000}


The 1/f noise is *reduced* of a factor about 10<sup>${reduction_factor_1f}</sup>.

*To estimate the uncertainty on the white noise level has been used the median deviation.


<h2>Spectra comparison</h2>

In this section spectra are compared in order to highlight possible similarities or discrepancies.

<h3>Demodolated and total power detector outputs</h3>

![](${polarimeter_name}_PSD_all_detector_outputs.svg){: class="plot"}


<h3>Stokes parameters signals</h3>

![](${polarimeter_name}_PSD_I_Q_U.svg){: class="plot"}

<h3>I signals</h3>

![](${polarimeter_name}_PSD_PWR0_Q1_PWR1_U1_PWR2_U2_PWR3_Q2_I.svg){: class="plot"}

<h3>Q signals</h3>

![](${polarimeter_name}_PSD_DEM0_Q1_DEM3_Q2_Q.svg){: class="plot"}


<h3>U signals</h3>

![](${polarimeter_name}_PSD_DEM1_U1_DEM2_U2_U.svg){: class="plot"}


<h3>Spectrograms</h3>

![](${polarimeter_name}_spectrogram_DEM.png){: class="plot"} 
![](${polarimeter_name}_spectrogram_PWR.png){: class="plot"}
![](${polarimeter_name}_spectrogram_IQU.png){: class="plot"}


