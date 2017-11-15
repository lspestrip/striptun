## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise characteristics for
the Strip polarimeter ${polarimeter_name}.

The report has been generated on ${date} using striptun v${striptun_version}
(commit `${latest_git_commit}`). 

This test has been performed with a sampling frequency of ${sampling_frequency} [Hz]
 and lasted ${'{0:.2f}'.format(test_duration)} [hours].

<h2>Results</h2>

The Power Spectral Densities (PSD) of the four detector outputs and of 
thier opportune combinations I, Q, U have been estimated using Welch's method as described in 
"Numerical recipes - The art of Scientific Computing", W.H. Press et al., Third edition
, pp.652-662. The function "signal.welch" of the Python library "scipy" has been used to 
implement it in the code.

The original data samples have been divided into ${n_chunks} segments of equal length. 
For each of them, the periodogram has been estimated and then the average at each 
frequency has been computed. Each segment has been detrended by subtracting a ${detrend} fit of the data.

<h2>White noise and 1/f estimation</h2>

The white noise level has been estimated calculating the median value of the spectrum starting 
from ${right_freq} [Hz]. 
The slope of the pink spectrum has been extracted with a linear fit of the data until 
${left_freq} [Hz]. 
The knee frequency has been estimated by doing the intersection between the median of the right
part of the spectrum and the linear fit of the left part of the spectrum.

<h3>Demodolated detector outputs</h3>

In this section are reported the results of the analysis for the four demodulated detector 
outputs.

![](${polarimeter_name}_PSD_DEM0_Q1.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM1_U1.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM2_U2.svg){: class="plot"} 
![](${polarimeter_name}_PSD_DEM3_Q2.svg){: class="plot"} 


DETECTOR  | f knee [mHz]         | alpha [#]           | white noise level [ADU^2/Hz] 
--------- |:--------------------:|:-------------------:|:----------------------:
DEM0/Q1   | ${'{0:.2f}'.format(DEM0Q1['f_knee'])} | ${'{0:.2f}'.format(DEM0Q1['slope'])} | ${'{0:.2f}'.format(DEM0Q1['WN_level'])} 
DEM1/U1   | ${'{0:.2f}'.format(DEM1U1['f_knee'])} | ${'{0:.2f}'.format(DEM1U1['slope'])} | ${'{0:.2f}'.format(DEM1U1['WN_level'])} 
DEM2/U2   | ${'{0:.2f}'.format(DEM2U2['f_knee'])} | ${'{0:.2f}'.format(DEM2U2['slope'])} | ${'{0:.2f}'.format(DEM2U2['WN_level'])} 
DEM3/Q2   | ${'{0:.2f}'.format(DEM3Q2['f_knee'])} | ${'{0:.2f}'.format(DEM3Q2['slope'])} | ${'{0:.2f}'.format(DEM3Q2['WN_level'])} 


<h3>Stokes parameters signals</h3>

In this section are reported the results of the analysis for the opportune combinations of the four 
detector outputs, which provides the Stokes parameters.

The following combinations of the detector outputs have been used:

I = (PWR0/Q1 + PWR1/U1 + PWR2/U2 + PWR3/Q2) / 4

Q = (DEM0/Q1 - DEM3/Q2) / 2 

U = (DEM1/U1 - DEM2/U2) / 2 

![](${polarimeter_name}_PSD_I.svg){: class="plot"}
![](${polarimeter_name}_PSD_Q.svg){: class="plot"}
![](${polarimeter_name}_PSD_U.svg){: class="plot"}

SIGNAL  | f knee [mHz]   | alpha [#]     | white noise level [ADU^2/Hz] 
------- |:--------------:|:-------------:|:----------------------:
I       | ${'{0:.2f}'.format(I['f_knee'])} | ${'{0:.2f}'.format(I['slope'])} | ${'{0:.2f}'.format(I['WN_level'])} 
Q       | ${'{0:.2f}'.format(Q['f_knee'])} | ${'{0:.2f}'.format(Q['slope'])} | ${'{0:.2f}'.format(Q['WN_level'])} 
U       | ${'{0:.2f}'.format(U['f_knee'])} | ${'{0:.2f}'.format(U['slope'])} | ${'{0:.2f}'.format(U['WN_level'])} 


The 1/f noise is reduced of a factor about ${'{0:1.0}'.format(reduction_1f)}.

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


